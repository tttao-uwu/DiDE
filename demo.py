import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
import argparse
import imageio
from PIL import Image
from dide.pipelines import TrellisImageTo3DPipeline
from dide.utils import render_utils, postprocessing_utils
import cv2
import numpy as np
import torch
from processing import turn_gray, palette_voronoi
# cache path

parser = argparse.ArgumentParser()

parser.add_argument('--cnt',   default="", type=str, help='Content image paths')

parser.add_argument('--sty',   default="",type=str, help='Texture image paths')

parser.add_argument('--color', default="", type=str, help='color image paths')


args = parser.parse_args()

def load_images(path):
    if os.path.isdir(path):
        return [Image.open(os.path.join(path, img)) for img in os.listdir(path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    elif os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return [Image.open(path)]
    else:
        raise ValueError(f"Invalid image path: {path}")

image_content = load_images(args.cnt)


image_texture = [turn_gray(load_images(args.sty))]

# NOTE: n_colors = the number of main colors extracted from the color image
image_color = [palette_voronoi(args.color, n_colors=3, k=5, width=518, height=518)]


content_name = os.path.splitext(os.path.basename(args.cnt))[0]
texture_name = os.path.splitext(os.path.basename(args.sty))[0]
color_name = os.path.splitext(os.path.basename(args.color))[0]
print(f"Content image: {content_name}, texture image: {texture_name}, Color image: {color_name}")

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

gamma = 1.1  # Increase brightness
inv_gamma = 1.0 / gamma
table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
table = np.array(table, np.uint8)

for i in range(len(image_texture)):
    image_texture[i] = Image.fromarray(cv2.LUT(np.array(image_texture[i]), table))


outputs = pipeline.run_multi_image(
    image_content,
    image_texture,
    image_color,
    seed=42,
    sparse_structure_sampler_params={
        "steps": 25,
        "cfg_strength": 3.0,
    },
    slat_sampler_params={
        "steps": 25,
        "cfg_strength": 3.0,
    },
)

video = render_utils.render_video(outputs['gaussian'][0])['color']

path = "./output"

if not os.path.exists(f"{path}"):
    os.makedirs(f"{path}")
    
imageio.mimsave(f"{path}/{content_name}+{texture_name}+{color_name}.mp4", video, fps=30)
