from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image

import torch


base_model_path = "./dataroot/models/runwayml/stable-diffusion-v1-5"
controlnet_path = "./output/checkpoint-4000/controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

control_image = load_image("./struct/hint/bjy_7_1_p1.png")
prompt = "High house in intensity 8.0"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=200, generator=generator, image=control_image
).images[0]
image.save("./output.png")
