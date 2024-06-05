# -*- coding: utf-8 -*-

import base64

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def base64_to_ndarray(b64_data: str):
    """base64转numpy数组

    Args:
        b64_data (str): base64数据

    Returns:
        _type_: _description_
    """
    image_bytes = base64.b64decode(b64_data)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


def bytes_to_ndarray(img_bytes: str):
    """字节转numpy数组

    Args:
        img_bytes (str): 图片字节

    Returns:
        _type_: _description_
    """
    image_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image_np2

def xyxyxyxy2xywh(bboxes):
    new_bboxes = np.zeros([len(bboxes), 4])
    new_bboxes[:, 0] = bboxes[:, 0::2].min()  # x1
    new_bboxes[:, 1] = bboxes[:, 1::2].min()  # y1
    new_bboxes[:, 2] = bboxes[:, 0::2].max() - new_bboxes[:, 0]  # w
    new_bboxes[:, 3] = bboxes[:, 1::2].max() - new_bboxes[:, 1]  # h
    return new_bboxes

def xyxy2xywh(bboxes):
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x center
    new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # y center
    new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
    new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
    return new_bboxes

def quad_coords_to_xyxy(quad_coords):
    x_values = [x for x, _ in quad_coords]
    y_values = [y for _, y in quad_coords]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Calculate the center (x_center, y_center) and width (w) and height (h)
    # x_center = (x_min + x_max) / 2
    # y_center = (y_min + y_max) / 2
    # w = x_max - x_min
    # h = y_max - y_min
    return x_min,y_min,x_max,y_max

def drawResult(img, boxes, txts):
    print("text: ", txts)
    d = ImageDraw.Draw(Image.fromarray(img))
    fnt = ImageFont.truetype('fonts/NotoSans-Regular.ttf', 5)
    for i in range(len(txts)):
        x_min,y_min,x_max,y_max = boxes[i]
        img = cv2.rectangle(img, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0, 255, 0), 2)
        img = cv2.putText(img, txts[i], (int(x_min),int(y_min)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        # info = txts[i]
        # info.encode("utf-8")
        # d.text([0,0], info, fill=(255, 0, 0), font=fnt)

    return img

