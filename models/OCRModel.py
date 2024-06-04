# -*- coding: utf-8 -*-

from typing import List, Set

from pydantic import BaseModel
from paddleocr import PaddleOCR, draw_ocr
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
from PIL import Image
import io
import cv2


class OCRModel(BaseModel):
    coordinate: List  # 图像坐标
    result: Set


class Base64PostModel(BaseModel):
    base64_str: str  # base64字符串


class ImageReader():

    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    def ProcessImage(self, imageFileBytes):
        img = bytes_to_ndarray(imageFileBytes)
        result = self.ocr.ocr(img=img, cls=False, rec=False)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)
        boxes = [line[0] for line in result]
        
        txts = []
        scores = []
        for i in range(len(boxes)):
            txts.append("i")
            scores.append(i)

        im_show = draw_ocr(img, boxes, txts, scores, font_path='fonts/NotoSans-Regular.ttf')
        cv2.imshow("im_show", im_show)
        cv2.waitKey()
        im_show = Image.fromarray(im_show)
        bytes_image = io.BytesIO()
        im_show.save(bytes_image, format='JPG')
        return bytes_image