import torch
from diffusers import StableDiffusionPipeline

# Load pretrained model
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

# Use CPU
pipe = pipe.to("cpu")

# Give text prompt
prompt = "A cat sitting on a chair"

# Generate image
image = pipe(prompt).images[0]

# Save image
image.save("generated_image.png")

print("Image generated successfully")