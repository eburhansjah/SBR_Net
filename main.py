import torch
import yaml
import time
from yaml import load
from src.SBR_NET import SBR_Net, kaiming_he_init

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

if __name__ == "__main__":
    start_time = time.time()

    # Reading values from config file
    stream = open("config.yaml", 'r')
    docs = list(yaml.load_all(stream, Loader))

    if docs:
        batch_size = docs[0].get("batch_size")
        in_channels_rfv = docs[0].get("in_channels_rfv")
        in_channels_stack = docs[0].get("in_channels_stack")
        num_blocks = docs[0].get("num_blocks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rfv_input = torch.randn(batch_size, in_channels_rfv, 224, 224).to(device)
    stack_input = torch.randn(batch_size, in_channels_stack, 224, 224).to(device)

    model = SBR_Net(in_channels_rfv=in_channels_rfv, 
                    in_channels_stack=in_channels_stack,
                    num_blocks=num_blocks).to(device)

    model.apply(kaiming_he_init)
    
    output = model(rfv_input, stack_input)
    print(output.shape) # Should be: Bx24x224x224

    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration of running the program: {duration: .2f} seconds")