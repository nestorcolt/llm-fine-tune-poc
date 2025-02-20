import torch

print(torch.__version__)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)

compute_dtype = getattr(torch, "float16")
print(compute_dtype)

x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
