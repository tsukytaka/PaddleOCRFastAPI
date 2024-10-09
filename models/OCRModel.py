# -*- coding: utf-8 -*-

from typing import List, Set

from pydantic import BaseModel
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
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='cpu')
        self.args, unknown = parser.parse_known_args()
        kwargs = {}
        self.model_plate_no = load_from_checkpoint('parseq_rec_model/parseq_plate_no_2024_09_13.ckpt', **kwargs).eval().to(self.args.device)
        self.model = load_from_checkpoint('parseq_rec_model/parseq_2024_09_23.ckpt', **kwargs).eval().to(self.args.device)
        self.model_writer_1 = load_from_checkpoint('parseq_rec_model/parseq_writer_1.ckpt', **kwargs).eval().to(self.args.device)
        
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)

    def ReadImageWithPos(self, imageFileBytes, configs, items):
        img = bytes_to_ndarray(imageFileBytes)
        orgImg = img.copy()
        drawImg = orgImg.copy()
        boxes = items[0]
        for i in range(len(boxes)):
            boxes[i] = (quad_coords_to_xyxy(boxes[i]))
        boxes = mergeLine(boxes)

        orgImg = img.copy()
        drawImg = orgImg.copy()
        txts = []
        images=[]
        origBoxes = []
        #rec by parseq
        for i in range(len(boxes)):
            x_min,y_min,x_max,y_max = boxes[i]
            w,h = x_max-x_min,y_max-y_min
            externRatio = 0.1
            x = max(0, x_min - int(w*externRatio*0.5))
            y = max(0, y_min - int(h*externRatio*0.5))
            w += int(w*externRatio)
            origBoxes.append([int(x),int(y),int((x + w)),int((y + h))])
            textImg = orgImg[origBoxes[i][1]:origBoxes[i][3], origBoxes[i][0]:origBoxes[i][2]]
            images.append(self.img_transform(Image.fromarray(textImg, 'RGB')))

        if len(images) > 0:
            images = torch.stack(images).to(self.args.device)
            with torch.no_grad():
                p = self.model(images)
                p =  torch.softmax(p, dim=2)
                p[:, :, 11:74] = 0
                p[:, :, 75:76] = 0
                p[:, :, 77:] = 0
                pred, p = self.model.tokenizer.decode(p)
            txts = pred

        result_txts = txts
        result_boxs = origBoxes
        if len(configs) > 0:
            result_txts = []
            result_boxs = []
            numberDigits = configs["total_digit"]
            numberDigitBeforeDot = configs["digit_before_dot"]
            for i in range(len(txts)):
                text = txts[i]
                text = re.sub("[\D]", "", text)
                if len(text) == numberDigits or numberDigits == 0:
                    if numberDigitBeforeDot > 0 and numberDigitBeforeDot < len(text):
                        text = text[:numberDigitBeforeDot] + '.' + text[numberDigitBeforeDot:]
                    result_txts.append(text)
                    result_boxs.append(origBoxes[i])
                
        drawImg = drawResult(drawImg, result_boxs, result_txts)
        array = cv2.cvtColor(np.array(drawImg), cv2.COLOR_RGB2BGR)
        im_show = Image.fromarray(array, mode="RGB")
        bytes_image = io.BytesIO()
        im_show.save(bytes_image, format='PNG')
        return bytes_image.getvalue()