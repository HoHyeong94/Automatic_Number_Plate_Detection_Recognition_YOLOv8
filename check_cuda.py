import torch
import torchvision
import torchaudio
import paddle

gpu_available  = paddle.device.is_compiled_with_cuda()
print("GPU available:", gpu_available)

print(torch.cuda.is_available())
print(torch.version.cuda)
print("PyTorch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("torchaudio version:", torchaudio.__version__)
