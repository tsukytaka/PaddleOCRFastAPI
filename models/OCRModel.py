# -*- coding: utf-8 -*-

from typing import List, Set

from pydantic import BaseModel
from paddleocr import PaddleOCR, draw_ocr
from utils.ImageHelper import *
from PIL import Image
import io
import cv2
import glob
import torch
import argparse
import re

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from tqdm import tqdm


class OCRModel(BaseModel):
    coordinate: List  # 图像坐标
    result: Set


class Base64PostModel(BaseModel):
    base64_str: str  # base64字符串

@torch.inference_mode()
class ImageReader():

    def __init__(self):
        # self.ocr = PaddleOCR(use_angle_cls=False, lang='japan')
        # self.ocr = PaddleOCR(use_angle_cls=False, lang='japan', rec_model_dir="./chalk_font_hwjp_number_PP-OCRv3_inference", rec_char_dict_path="./chalk_font_hwjp_number_PP-OCRv3_inference/dict.txt")
        parser = argparse.ArgumentParser()
        # parser.add_argument('--checkpoint', default='parseq_rec_model/parseq-2024_05_19.ckpt' , help="Model checkpoint (or 'pretrained=<model_id>')")
        # parser.add_argument('--checkpoint', default='parseq_rec_model/best-2024-06-11.ckpt' , help="Model checkpoint (or 'pretrained=<model_id>')")
        
        # parser.add_argument('--images', nargs='+', help='Images to read')
        parser.add_argument('--device', default='cpu')
        self.args, unknown = parser.parse_known_args()
        kwargs = {} #parse_model_args(unknown)
        # kwargs["model"] = dict()
        # kwargs['model']['charset_test'] = "0123456789"
        # print(kwargs)
        # print(f'Additional keyword arguments: {kwargs}')
        self.model_plate_no = load_from_checkpoint('parseq_rec_model/parseq_plate_no_2024_09_13.ckpt', **kwargs).eval().to(self.args.device)
        self.model = load_from_checkpoint('parseq_rec_model/parseq_2024_09_23.ckpt', **kwargs).eval().to(self.args.device)
        self.model_writer_1 = load_from_checkpoint('parseq_rec_model/parseq_writer_1.ckpt', **kwargs).eval().to(self.args.device)
        self.model_text_value = load_from_checkpoint('parseq_rec_model/parseq_text_value_2024_10_15.ckpt', **kwargs).eval().to(self.args.device)
        
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
    
    def postprocess_prediction(self, prediction, part3_max_length):
        """Postprocess the model's prediction."""
        if prediction[0] == "D":
            formats = [7, 2, part3_max_length]
            parts = prediction.split('-')
            
            if len(parts) != 3:
                parts = [parts[0], parts[1][:2], parts[1][2:]]
            
            for i in range(1, len(formats)):
                parts[i] = parts[i][:formats[i]]
            
            parts[1] = parts[1].replace('/', '1')
            parts[1] = parts[1].replace('\\', '1')
            parts[2] = parts[2].replace('/', '1') if len(parts[2]) == 3 else parts[2]
            parts[2] = parts[2].replace('\\', '1') if len(parts[2]) == 3 else parts[2]
            
            return "-".join(parts)
        return prediction

    def ReadImageWithPos(self, imageFileBytes, items, modelType):
        img = bytes_to_ndarray(imageFileBytes)
        orgImg = img.copy()
        drawImg = orgImg.copy()
        txts = []
        scores = [] 
        images=[]
        origBoxes = []
        plateNoImgs = []
        textValueImgs = []
        #rec by parseq
        plateNoItems = []
        valueItems = []
        textValueItems = []
        for i in range(len(items)):
            x,y,w,h = items[i]["position"]
            origBoxes.append([x,y,x+w,y+h])
            textImg = orgImg[origBoxes[i][1]:origBoxes[i][3], origBoxes[i][0]:origBoxes[i][2]]
            if items[i]["title"] == "plateNo":
                plateNoImgs.append(self.img_transform(Image.fromarray(textImg, 'RGB')))
                plateNoItems.append(items[i])
            elif items[i]["modeData"] == 2:
                if False:
                    outputImg = Image.fromarray(textImg, 'RGB')
                    outputImg.save("cropTextImgs/{index}.png".format(index = i))
                textValueImgs.append(self.img_transform(Image.fromarray(textImg, 'RGB')))
                textValueItems.append(items[i])
            else:
                images.append(self.img_transform(Image.fromarray(textImg, 'RGB')))
                valueItems.append(items[i])

        if len(plateNoImgs) > 0:
            plateNoImgs = torch.stack(plateNoImgs).to(self.args.device)
            with torch.no_grad():
                p = self.model_plate_no(plateNoImgs)
                p =  torch.softmax(p, dim=2)
                pred, p = self.model_plate_no.tokenizer.decode(p)
                txts = pred
                scores = ([s.cpu().mean().item() for s in p])
                for i in range(len(txts)):
                    txt = txts[i]
                    txt = self.postprocess_prediction(txt,9)
                    txt = re.sub("[,.]", "", txt)
                    plateNoItems[i]["content"] =txt
                    plateNoItems[i]["score"] =scores[i]

        if len(textValueImgs) > 0:
            textValueImgs = torch.stack(textValueImgs).to(self.args.device)
            with torch.no_grad():
                p = self.model_text_value(textValueImgs)
                p =  torch.softmax(p, dim=2)
                pred, p = self.model_text_value.tokenizer.decode(p)
                txts = pred
                scores = ([s.cpu().mean().item() for s in p])
                for i in range(len(txts)):
                    txt = txts[i]
                    # txt = self.postprocess_prediction(txt,9)
                    # txt = re.sub("[,.]", "", txt)
                    textValueItems[i]["content"] =txt
                    textValueItems[i]["score"] =scores[i]

        if len(images) > 0:
            images = torch.stack(images).to(self.args.device)
            with torch.no_grad():
                p = self.model(images)
                p =  torch.softmax(p, dim=2)
                p[:, :, 11:74] = 0
                p[:, :, 75:76] = 0
                p[:, :, 77:] = 0
                if modelType == 0:
                    pred, p = self.model.tokenizer.decode(p)
                elif modelType == 1:
                    pred, p = self.model_writer_1.tokenizer.decode(p)
                txts = pred
                scores = ([s.cpu().mean().item() for s in p])
                for i in range(len(txts)):
                    txt = txts[i]
                    txt = re.sub(",", ".", txt)
                    txt = re.sub("(^\\D+)","", txt)
                    txt = re.sub("(\\D+$)","", txt)
                    valueItems[i]["content"] =txt
                    valueItems[i]["score"] =scores[i]


        drawImg = drawResult(drawImg, origBoxes, txts)
        array = cv2.cvtColor(np.array(drawImg), cv2.COLOR_RGB2BGR)
        im_show = Image.fromarray(array, mode="RGB")
        bytes_image = io.BytesIO()
        im_show.save(bytes_image, format='PNG')
        items = plateNoItems + textValueItems + valueItems
        return items, bytes_image.getvalue()