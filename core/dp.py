import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        B, N, D = X.size()
        p = torch.arange(N, dtype=torch.float32).view(-1, 1) / \
            torch.pow(10000, torch.arange(0, D, 2, dtype=torch.float32) / D)
        P = torch.zeros((B, N, D))
        P[:, :, 0::2] = torch.sin(p)
        P[:, :, 1::2] = torch.cos(p)
        return self.dropout(X + P.to(X.device))

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    """
    Q: n * d
    K: m * d
    V: m * v
    Q * K_T * V
    """
    def forward(self, queries, keys, values):
        d = queries.size()[-1]
        attention_weights = nn.functional.softmax(
            torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d), dim = -1)
        scores = torch.bmm(attention_weights, values)
        return scores


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_input, ffn_hidden, ffn_output, activate_fun = "relu", **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_input, ffn_hidden)
        if "relu" == activate_fun:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.GELU()
        self.dense2 = nn.Linear(ffn_hidden, ffn_output)

    def forward(self, X):
        return self.dense2(self.activate(self.dense1(X)))


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        query_size, 
        key_size, 
        value_size, 
        feature_dims, 
        num_heads, 
        dropout, 
        bias = False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, feature_dims, bias = bias)
        self.W_k = nn.Linear(key_size, feature_dims, bias = bias)
        self.W_v = nn.Linear(value_size, feature_dims, bias = bias)
        self.W_o = nn.Linear(feature_dims, feature_dims, bias = bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class Deformation_Layer(nn.Module):
    def __init__(
        self,
        dropout,
        num_heads,
        fea_len,
        bias,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
                            fea_len, 
                            fea_len, 
                            fea_len, 
                            fea_len, 
                            num_heads, 
                            dropout, 
                            bias
                        )
        self.addnorm = AddNorm(fea_len, dropout)
        self.fc = PositionWiseFFN(fea_len, fea_len * 2, fea_len, "relu")
        self.addnormfc = AddNorm(fea_len, dropout)

    def forward(self, x, y, z):
        y = self.addnorm(z, self.attention(x, y, z))
        y = self.addnormfc(z, self.fc(y))
        return y


class Deformation_Module(nn.Module):
    def __init__(
        self, 
        num_layers = 1, 
        dropout = 0.5, 
        num_heads = 8, 
        fea_len = 64, 
        bias = False
    ):
        super().__init__()
        self.fea_len = fea_len
        self.pos_encoding = PositionalEncoding(dropout)
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(
                "Deformation_Layer" + str(i),
                Deformation_Layer(
                        dropout, num_heads, fea_len, bias
                    )
                )
    def forward(self, x, y, z):
        x = self.pos_encoding(x * math.sqrt(self.fea_len))
        y = self.pos_encoding(y * math.sqrt(self.fea_len))
        z = self.pos_encoding(z * math.sqrt(self.fea_len))
        for layer in self.layers:
            y = layer(x, y, z)
        return y

class Loading_Module(nn.Module):
    def __init__(
        self, 
        num_layers = 2, 
        dropout = 0, 
        num_heads = 8, 
        fea_len = 64, 
        bias = False
    ):
        super().__init__()
        self.loading_module = Deformation_Module(
                                    num_layers,
                                    dropout,
                                    num_heads,
                                    fea_len,
                                    bias
                                )
    def forward(self, params, strip, mould):
        return self.loading_module(params, strip, mould)


class Unloading_Module(nn.Module):
    def __init__(
        self, 
        dropout = 0, 
        num_heads = 8, 
        fea_len = 64, 
        bias = False
    ):
        super().__init__()
        self.unloading_module = Deformation_Module(
                                    1,
                                    dropout,
                                    num_heads,
                                    fea_len,
                                    bias
                                )

    def forward(self, strip):
        return self.unloading_module(strip, strip, strip)

if __name__ == "__main__":
    x = torch.randn(30, 64, 64)
    y = torch.randn(30, 64, 64)
    z = torch.randn(30, 64, 64)

    loading_module = Loading_Module()

    output_loading = loading_module(x, y, z)

    unloading_module = Unloading_Module()

    output_springback = unloading_module(output_loading)
