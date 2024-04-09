import torch
import torch.nn as nn
from .cle_and_clr import Characteristic_Line_Extractor, Characteristic_Line_Reverser
from .cse_and_csr import Cross_Section_Extractor, Cross_Section_Reverser, Cross_Section_Reshaper
from .off import Object_Feature_Fusioner
from .mpe import Motion_Parameters_Extractor
from .dp import Loading_Module, Unloading_Module

class LaDEEP(nn.Module):
    def __init__(self):
        super().__init__()
        self.cle_for_mould = Characteristic_Line_Extractor()
        self.cle_for_strip = Characteristic_Line_Extractor()
        self.csr = Cross_Section_Reverser()
        self.cse = Cross_Section_Extractor()
        self.csr_mlp = Cross_Section_Reshaper()
        self.off = Object_Feature_Fusioner()
        self.mpe = Motion_Parameters_Extractor()
        self.dp_loading = Loading_Module()
        self.dp_unloading = Unloading_Module()
        self.clr = Characteristic_Line_Reverser()
        self._initialize()

    def forward(self, strip, mould, section, params):
        mould = self.cle_for_mould(mould)
        strip = self.cle_for_strip(strip)
        section = self.cse(section)
        recovery = self.csr(section)
        section = self.csr_mlp(section)
        strip = self.off(strip, section)
        params = self.mpe(params)
        strip = self.dp_loading(params, strip, mould)
        strip = self.dp_unloading(strip)
        strip = self.clr(strip)

        return strip, recovery
    
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain = nn.init.calculate_gain("relu"))
        return True

if __name__ == "__main__":
    mould = torch.randn(32, 3, 300)
    strip = torch.randn(32, 3, 300)
    section = torch.randn(32, 1, 512, 256)
    params = torch.randn(32, 1, 6)

    net = LaDEEP()

    output_strip, output_recovery = net(strip, mould, section, params)
