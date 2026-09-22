import torch

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)

    print("GPU:", props.name)
    print("VRAM:", round(props.total_memory / 1024**3, 2), "GB")