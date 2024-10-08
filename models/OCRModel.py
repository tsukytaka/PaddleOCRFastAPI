# -*- coding: utf-8 -*-

from typing import List, Set

from pydantic import BaseModel
from paddleocr import PaddleOCR, draw_ocr
from utils.ImageHelper import *
from PIL import Image
import io
import cv2
import re


class OCRModel(BaseModel):
    coordinate: List  # 图像坐标
    result: Set


class Base64PostModel(BaseModel):
    base64_str: str  # base64字符串


class ImageReader():

    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=False, lang='japan', rec_model_dir="./chalk_font_hwjp_number_PP-OCRv3_inference", rec_char_dict_path="./chalk_font_hwjp_number_PP-OCRv3_inference/dict.txt")
        

    def DetectTextBox(self, imageFileBytes):
        img = bytes_to_ndarray(imageFileBytes)
        formRatio = 480.0 / img.shape[1]
        img = cv2.resize(img, (0,0), fx=formRatio, fy=formRatio)
        result = self.ocr.ocr(img=img, cls=False, rec=False)
        for i in range(len(result)):
            boxes = result[i]
            for j in range(len(boxes)):
                box = boxes[j]
                for k in range(len(box)):
                    point = box[k]
                    point[0] /= formRatio
                    point[1] /= formRatio
                    box[k] = point
                boxes[j] = box
            result[i] = boxes
 
        print("result: ", result)
        return result
    
    def ReadImageWithPos(self, imageFileBytes, configs, items):
        img = bytes_to_ndarray(imageFileBytes)
        orgImg = img.copy()
        drawImg = orgImg.copy()
        boxes = items[0]
        for i in range(len(boxes)):
            boxes[i] = (quad_coords_to_xyxy(boxes[i]))
        boxes = mergeLine(boxes)

        txts = []
        origBoxes = []
        for i in range(len(boxes)):
            x_min,y_min,x_max,y_max = boxes[i]
            w,h = x_max-x_min,y_max-y_min
            externRatio = 0.1
            x = max(0, x_min - int(w*externRatio*0.5))
            y = max(0, y_min - int(h*externRatio*0.5))
            w += int(w*externRatio)
            origBoxes.append([int(x),int(y),int((x + w)),int((y + h))])
            textImg = orgImg[origBoxes[i][1]:origBoxes[i][3], origBoxes[i][0]:origBoxes[i][2]]
            
            grayImg = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)
            T, binImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            textImg = binImg
            result = self.ocr.ocr(img=textImg, cls=False, det=False)
            print("result: ", result)
            txts.append(result[0][0][0])
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