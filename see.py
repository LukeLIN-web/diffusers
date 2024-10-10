from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D,CrossAttnDownBlock2D

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

# model = pipe.unet
# model = pipe.vae
model = pipe.unet

# unet_debugging = torch.load("unet.pth")

def get_module_param_size(module):
    param_size = 0
    for param in module.parameters():
        param_size += param.numel()  # 获取参数个数
    return param_size

def print_block_params(model):
    for name, module in model.named_modules():
        print(f"Module name: {name}")
        if isinstance(module, CrossAttnUpBlock2D):  # 替换为你的 Cross-Attention 类
            param_size = get_module_param_size(module)
            print(f"Cross-Attention Block '{name}': {param_size} parameters")
        
        if isinstance(module, ResnetBlock2D):  # 替换为你的 ResNetBlock 类
            param_size = get_module_param_size(module)
            print(f"ResNet Block '{name}': {param_size} parameters")




# vae = torch.load("vae.pth")

# print(unet_debugging)
# print(vae)