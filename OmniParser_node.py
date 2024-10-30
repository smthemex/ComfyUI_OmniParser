# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import base64, os
import io
import random
from PIL import Image
import torch
import logging
import easyocr

import folder_paths
from .utils import get_config,load_list_images,convert_safetensor_to_pt,tensor2imglist
from .OmniParser.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

node_cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

class OmniParser_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo": ("STRING", {"default": "microsoft/OmniParser", }),
                "platform": (["pc", "web","mobile"],),
            },

        }

    RETURN_TYPES = ("OP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main"
    CATEGORY = "OmniParser"

    def main(self, repo,platform,):
        if repo:
           if "florence" in repo.lower():
               model_name="florence2"
           elif "blip" in repo.lower():
               model_name="blip2"
           elif  repo== "microsoft/OmniParser":
               model_name = "default"
           else:
               raise "unsupport repo id"
        else:
            raise "must fill repo or local path"
        
        yolo_model_converted=os.path.join(node_cur_path,"OmniParser/weights/icon_detect/best.pt")
        if not os.path.exists(yolo_model_converted):
            convert_safetensor_to_pt()
            logging.info("convert_safetensor_to_pt is  done.")

        yolo_model = get_yolo_model(model_path=yolo_model_converted)
        
        caption_model_processor = get_caption_model_processor(model_name=model_name,
                                                              model_name_or_path=repo)
        draw_bbox_config=get_config(platform)
        
        logging.info("loading checkpoint done.")
        model={"yolo_model":yolo_model,"caption_model_processor":caption_model_processor,"draw_bbox_config":draw_bbox_config}
        return (model,)
    
class OmniParser_Sampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":("OP_MODEL",),
                "image": ("IMAGE",),  # [B,H,W,C], C=3
                "text_threshold":("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "number",}),
                "box_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number", }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number", }),
                "orc_model": (["easyocr", "paddleocr"],),
                "orc_mode": (["en", "'ch_sim','en'"],),
                "output_bb_format": (["xyxy", "xywh"],),
                "display_img": ("BOOLEAN", {"default": False},),
                "paragraph": ("BOOLEAN", {"default": False},),
                "output_coord_in_ratio": ("BOOLEAN", {"default": True},),
            },
        }
    
    RETURN_TYPES = ("IMAGE","STRING")
    RETURN_NAMES = ("image","string")
    FUNCTION = "main"
    CATEGORY = "OmniParser"
    
    def main(self, model,image,text_threshold,box_threshold,iou_threshold,orc_model,orc_mode,output_bb_format,display_img,paragraph,output_coord_in_ratio,**kwargs):
        #optional chocie
        optional_c=True
        
        #pre model
        yolo_model=model.get("yolo_model")
        draw_bbox_config=model.get("draw_bbox_config")
        caption_model_processor=model.get("caption_model_processor")
        
        if orc_model=="easyocr":
            ORC = easyocr.Reader([orc_mode])
            use_paddleocr = False
        else:
            use_paddleocr = True #another choice
            from paddleocr import PaddleOCR
            if orc_mode=="'ch_sim','en'":
                orc_mode="ch"
            ORC = PaddleOCR(
                lang=orc_mode,  # other lang also available
                use_angle_cls=False,
                use_gpu=False,  # using cuda will conflict with pytorch in the same process
                show_log=False,
                max_batch_size=1024,
                use_dilation=True,  # improves accuracy
                det_db_score_mode='slow',  # improves accuracy
                rec_batch_num=1024)
            
        #pre image if list
        file_prefix = ''.join(random.choice("0123456789abcdefg") for _ in range(6))
        image_list,B = tensor2imglist(image, np_out=True)
        path_list = []
        if B==1:
            image_input = image_list[0]
            image_save_path = os.path.join(folder_paths.get_input_directory(), f'Temp_{file_prefix}.png')
            image_input.save(image_save_path)
            path_list=[image_save_path]
        else:
            for i,img in enumerate(image_list):
                image_save_path = os.path.join(folder_paths.get_input_directory(), f'Temp_{file_prefix}_{i}.png')
                img.save(image_save_path)
                path_list.append(image_save_path)
        
        ouput_img=[]
        ouput_text=[]
        for batch_num,image_path in enumerate(path_list):
            # pre box
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, ORC, use_paddleocr,
                                                            display_img=display_img, output_bb_format=output_bb_format,
                                                            goal_filtering=None,
                                                            easyocr_args={'paragraph': paragraph,
                                                                          'text_threshold': text_threshold})
            text, ocr_bbox = ocr_bbox_rslt
            # infer
            logging.info("start infer..")
            dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(optional_c, image_path,
                                                                                          yolo_model,batch_num,
                                                                                          BOX_TRESHOLD=box_threshold,
                                                                                          output_coord_in_ratio=output_coord_in_ratio,
                                                                                          ocr_bbox=ocr_bbox,
                                                                                          draw_bbox_config=draw_bbox_config,
                                                                                          caption_model_processor=caption_model_processor,
                                                                                          ocr_text=text,
                                                                                          iou_threshold=iou_threshold)
            image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
            ouput_img.append(image)
            content_list = '\n'.join(parsed_content_list)
            ouput_text.append(content_list)
        print('finish processing')
        image=load_list_images(ouput_img)
        prompt='\n\n'.join(ouput_text)
        return (image,prompt)


NODE_CLASS_MAPPINGS = {
    "OmniParser_Loader": OmniParser_Loader,
    "OmniParser_Sampler":OmniParser_Sampler,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniParser_Loader": "OmniParser_Loader",
    "OmniParser_Sampler":"OmniParser_Sampler",
}
