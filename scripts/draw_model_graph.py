import os
import sys

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend([os.getcwd()])

from core.ladeep import LaDEEP
from utils.tensorboard_logging import Tensorboard_Logging
import torch

input_tensors = [
    torch.randn(8, 3, 300),
    torch.randn(8, 3, 300),
    torch.randn(8, 1, 512, 256),
    torch.randn(8, 1, 6)
]

model = LaDEEP()

tl_writer = Tensorboard_Logging("./logs/model")
tl_writer.write_model_graph(model, input_tensors)
tl_writer.writer_close()
