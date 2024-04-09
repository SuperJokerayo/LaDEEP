import torch
import torch.nn as nn

class Object_Feature_Fusioner(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = 1 / 2

        self.modes1, self.modes2 = 16, 16

        self.complex_fusion_operator1 = nn.Parameter(
            self.scale * torch.rand(2, 1, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.complex_fusion_operator2 = nn.Parameter(
            self.scale * torch.rand(2, 1, self.modes1, self.modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, feature_x, feature_y):
        feature_xy = torch.stack((feature_x, feature_y), dim = 1)

        b = feature_xy.shape[0]
        xy_ft = torch.fft.rfft2(feature_xy)

        out_ft = torch.zeros(
                    b, 
                    1,  
                    feature_xy.size(-2), 
                    feature_xy.size(-1)//2 + 1, 
                    dtype=torch.cfloat, 
                    device=feature_xy.device
                )
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(xy_ft[:, :, :self.modes1, :self.modes2], self.complex_fusion_operator1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(xy_ft[:, :, -self.modes1:, :self.modes2], self.complex_fusion_operator2)

        feature_xy = torch.fft.irfft2(out_ft, s=(feature_xy.size(-2), feature_xy.size(-1)))
        return feature_xy.squeeze(1)

if __name__ == "__main__":
    a = torch.randn(2, 64, 64)
    b = torch.randn(2, 64, 64)

    model = Object_Feature_Fusioner()
    print(model(a, b).shape)
