# !/usr/bin/env python
# -*- coding: UTF-8 -*-
from ultralytics.nn.tasks import DetectionModel
from safetensors.torch import load_file
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import torch
from comfy.utils import common_upscale,ProgressBar
import folder_paths

cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

POINT_SIZE = 4 # Size of the query point in the preview video


def convert_safetensor_to_pt():
    tensor_dict = load_file(os.path.join(cur_path,"OmniParser/weights/icon_detect/model.safetensors"))
    
    model = DetectionModel(os.path.join(cur_path,'OmniParser/weights/icon_detect/model.yaml'))
    model.load_state_dict(tensor_dict)
    torch.save({'model': model}, os.path.join(cur_path,'OmniParser/weights/icon_detect/best.pt'))


def get_config(platform="pc"):
    #platform = 'pc'
    if platform == 'pc':
        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 2,
            'thickness': 2,
        }
    elif platform == 'web':
        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 3,
            'thickness': 3,
        }
    elif platform == 'mobile':
        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 3,
            'thickness': 3,
        }
    else:
        raise "error"
    return draw_bbox_config

def spilit_tensor2list(img_tensor):#[B,H,W,C], C=3,B>=1
    video_list = []
    if isinstance(img_tensor, list):
        if isinstance(img_tensor[0], torch.Tensor):
            video_list = img_tensor
    elif isinstance(img_tensor, torch.Tensor):
        b, _, _, _ = img_tensor.size()
        if b == 1:
            img = [b]
            while img is not []:
                video_list += img
        else:
            video_list = torch.chunk(img_tensor, chunks=b)
    return video_list
    
def tensor2imglist(image,np_out=True):# pil first
    B, _, _, _ = image.size()
    if B == 1:
        if np_out:
            list_out = [tensor2pil(image)]
        else:
            list_out = [tensor2cv(image.squeeze())]
    else:
        image_list = torch.chunk(image, chunks=B)
        if  np_out:
            list_out = [tensor2pil(i) for i in image_list]
        else:
            list_out = [tensor2cv(i.squeeze()) for i in image_list]
    return list_out,B


def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def pil2narry(img):
    narry = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return narry

def tensor_upscale2pil(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor2pil(samples)
    return img_pil

def tensor_upscale(img_tensor, width, height): #torch tensor
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def tensor2cv(tensor_image):
    if tensor_image.is_cuda:
        tensor_image = tensor_image.detach().cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def cvargb2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def cv2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def images_generator(img_list: list,):
    #get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_,Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_,np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in=img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in,np.ndarray):
            i=cv2.cvtColor(img_in,cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            print(i.shape)
            return i
        else:
           raise "unsupport image list,must be pil,cv2 or tensor!!!"
        
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image

def load_list_images(img_list: list,):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

