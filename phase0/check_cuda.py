import torch
print("CUDA available:", torch.cuda.is_available())
print("torch version:", torch.__version__)
print("torchaudio version:", end=" ")
try:
    import torchaudio
    print(torchaudio.__version__)
except:
    print("not installed")
print("torchcodec:", end=" ")
try:
    import torchcodec
    print(torchcodec.__version__)
except:
    print("not installed")
