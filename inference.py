from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image

import torch


base_model_path = "./dataroot/models/runwayml/stable-diffusion-v1-5"
controlnet_path = "./controlnet-lora-output"
lora_path = "./controlnet-lora-output/pytorch_lora_weights.safetensors"
control_image_path = "./dataset/hint/bjy_7_1_p1.png"
prompt = "High house in intensity 8.0"


if __name__ == "__main__":
    # load model
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.unet.load_attn_procs(lora_path)

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # remove following line if xformers is not installed or when using Torch 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    # memory optimization
    pipe.enable_model_cpu_offload()

    # load control image
    control_image = load_image(control_image_path)

    # generate image
    generator = torch.manual_seed(100)
    image = pipe(
        prompt, num_inference_steps=200, generator=generator, image=control_image
    ).images[0]

    # post process
    a, b = control_image.size
    for i in range(a):
        for j in range(b):
            pixel = control_image.getpixel((i, j))
            if pixel != (132, 132, 132):
                image.putpixel((i, j), pixel)
            else:
                pixel = image.getpixel((i, j))
                if pixel[0] > 193 and pixel[1] < 62 and pixel[2] < 62:
                    image.putpixel((i, j), (255, 0, 0))
                else:
                    image.putpixel((i, j), (132, 132, 132))

    image.save("./output.png")
