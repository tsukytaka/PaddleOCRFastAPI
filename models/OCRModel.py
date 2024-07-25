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
        self.ocr = PaddleOCR(use_angle_cls=False, lang='japan', rec_model_dir="./chalk_font_hwjp_number_PP-OCRv3_inference", rec_char_dict_path="./chalk_font_hwjp_number_PP-OCRv3_inference/dict.txt")
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', default='parseq_rec_model/parseq-2024_05_19.ckpt' , help="Model checkpoint (or 'pretrained=<model_id>')")
        # parser.add_argument('--checkpoint', default='parseq_rec_model/best-2024-06-11.ckpt' , help="Model checkpoint (or 'pretrained=<model_id>')")
        
        # parser.add_argument('--images', nargs='+', help='Images to read')
        parser.add_argument('--device', default='cpu')
        self.args, unknown = parser.parse_known_args()
        kwargs = {} #parse_model_args(unknown)
        kwargs["model"] = dict()
        kwargs['model']['charset_test'] = "0123456789"
        print(kwargs)
        print(f'Additional keyword arguments: {kwargs}')

        self.model = load_from_checkpoint(self.args.checkpoint, **kwargs).eval().to(self.args.device)
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)

    def ProcessImage(self, imageFileBytes, modelType):
        img = bytes_to_ndarray(imageFileBytes)
        orgImg = img.copy()
        formRatio = 480.0 / img.shape[1]
        img = cv2.resize(img, (0,0), fx=formRatio, fy=formRatio)
        # npImg = Image.fromarray(img)
        drawImg = orgImg.copy()
        #detect by paddle
        result = self.ocr.ocr(img=img, cls=False, rec=False)
        print("result: ", result)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)
        boxes = result[0]
        print("boxes: ", boxes)
        print("size: ", len(boxes))
        for i in range(len(boxes)):
            boxes[i] = (quad_coords_to_xyxy(boxes[i]))
        boxes = mergeLine(boxes)
        txts = []
        scores = []
        images=[]
        origBoxes = []
        #rec by parseq
        if modelType == 1:
            for i in range(len(boxes)):
                x_min,y_min,x_max,y_max = boxes[i]
                w,h = x_max-x_min,y_max-y_min
                externRatio = 0.1
                x = max(0, x_min - int(w*externRatio*0.5))
                y = max(0, y_min - int(h*externRatio*0.5))
                w += int(w*externRatio)
                h += int(h*externRatio)
                origBoxes.append([int(x/formRatio),int(y/formRatio),int((x + w)/formRatio),int((y + h)/formRatio)])
                # origBoxes.append([int(x_min/formRatio),int(y_min/formRatio),int(x_max/formRatio),int(y_max/formRatio)])
                # drawImg = cv2.rectangle(drawImg, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0, 255, 0), 2)
                textImg = orgImg[origBoxes[i][1]:origBoxes[i][3], origBoxes[i][0]:origBoxes[i][2]]
                images.append(self.img_transform(Image.fromarray(textImg, 'RGB')))

                # # Load image and prepare for input
                # image = textImg.convert('RGB')
                # image = self.img_transform(image).unsqueeze(0).to(self.args.device)

                # p = self.model(image).softmax(-1)
                # pred, p = self.model.tokenizer.decode(p)
                # print(f'text: {pred[0]}')
                # txts.append(pred[0])
                # scores.append(p[0].cpu().mean().item())

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
                scores = ([s.cpu().mean().item() for s in p])
                # for i in range(len(txts)):
                #     if txts[i] == "4900" and orgImg.shape == (823, 1147, 3):
                #         txts[i] = "4900.4"
                #         textBox = origBoxes[i]
                #         textBox[1] -= int((textBox[3] - textBox[1])*0.05)
                #         textBox[3] += int((textBox[3] - textBox[1])*0.1)
                #         textBox[2] += int((textBox[2] - textBox[0])*0.25)
                #         origBoxes[i] = textBox
                #         # textImg = orgImg[textBox[1]:textBox[3], textBox[0]:textBox[2]]
                #         # image = self.img_transform(Image.fromarray(textImg, 'RGB'))
                #         # with torch.no_grad():
                #         #     p = self.model(torch.stack([image]).to(self.args.device))
                #         #     p =  torch.softmax(p, dim=2)
                #         #     p[:, :, 11:74] = 0
                #         #     p[:, :, 75:76] = 0
                #         #     p[:, :, 77:] = 0
                #         #     pred, p = self.model.tokenizer.decode(p)
                #         #     txts[i] = pred[0]
                #         #     scores[i] = p[0].cpu().mean().item()
                #         break
        elif modelType==2:
            for i in range(len(boxes)):
                x_min,y_min,x_max,y_max = boxes[i]
                w,h = x_max-x_min,y_max-y_min
                externRatio = 0.1
                x = max(0, x_min - int(w*externRatio*0.5))
                y = max(0, y_min - int(h*externRatio*0.5))
                w += int(w*externRatio)
                # h += int(h*externRatio)
                origBoxes.append([int(x/formRatio),int(y/formRatio),int((x + w)/formRatio),int((y + h)/formRatio)])
                textImg = orgImg[origBoxes[i][1]:origBoxes[i][3], origBoxes[i][0]:origBoxes[i][2]]
                
                grayImg = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)
                T, binImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                textImg = binImg#cv2.cvtColor(binImg, cv2.COLOR_GRAY2RGB)

                cv2.imwrite("tmp.png", textImg)

                result = self.ocr.ocr(img=textImg, cls=False, det=False)
                print("result: ", result)
                txts.append(result[0][0][0])


        #     cv2.putText(drawImg, txts[i], boxes[i][1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # im_show = draw_ocr_box_txt(img, boxes, txts, scores, font_path='fonts/NotoSans-Regular.ttf')
        # im_show = drawImg.copy()
        drawImg = drawResult(drawImg, origBoxes, txts)
        array = cv2.cvtColor(np.array(drawImg), cv2.COLOR_RGB2BGR)
        im_show = Image.fromarray(array, mode="RGB")
        bytes_image = io.BytesIO()
        im_show.save(bytes_image, format='PNG')
        return bytes_image.getvalue()