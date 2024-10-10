from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import matplotlib.pyplot as plt

# 加载模型
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

# 设置提示和生成参数
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
guidance_scale = 7.5  # Set the guidance scale to your desired value

# 生成图像
image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=guidance_scale).images[0]

# 确保图像已经生成
if isinstance(image, Image.Image):
    print("Image generated successfully!")

# 使用matplotlib显示并保存图像
plt.imshow(image)
plt.axis('off')  # 关闭坐标轴
plt.savefig("output_image.png", bbox_inches='tight', pad_inches=0)
plt.show()  # 显示图像
