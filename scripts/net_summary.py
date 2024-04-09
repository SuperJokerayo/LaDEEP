import os
import sys
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend([os.getcwd()])

from core.ladeep import LaDEEP
from torchinfo import summary


model = LaDEEP()

summary(model, [(2, 3, 300), (2, 3, 300), (2, 1, 512, 256), (2, 1, 6)], device = "cpu")