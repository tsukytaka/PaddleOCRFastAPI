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
        self.ocr = PaddleOCR(use_angle_cls=False, lang='japan', det_model_dir="./paddle_models/det/red_chalk_PP-OCR_v3_det_inference/Student", rec_model_dir="./chalk_font_hwjp_number_PP-OCRv3_inference", rec_char_dict_path="./chalk_font_hwjp_number_PP-OCRv3_inference/dict.txt")
        parser = argparse.ArgumentParser()
        
        # parser.add_argument('--images', nargs='+', help='Images to read')
        parser.add_argument('--device', default='cpu')
        self.args, unknown = parser.parse_known_args()
        kwargs = {} #parse_model_args(unknown)
        # kwargs["model"] = dict()
        # kwargs['model']['charset_test'] = "0123456789"
        # print(kwargs)
        # print(f'Additional keyword arguments: {kwargs}')
        # self.model_plate_no = load_from_checkpoint('parseq_rec_model/parseq_plate_no_2024_09_13.ckpt', **kwargs).eval().to(self.args.device)
        self.model = load_from_checkpoint('parseq_rec_model/parseq-2024_05_19.ckpt', **kwargs).eval().to(self.args.device)
        # self.model_writer_1 = load_from_checkpoint('parseq_rec_model/parseq_writer_1.ckpt', **kwargs).eval().to(self.args.device)
        # print(f'model_writer_1: parseq_rec_model/parseq_writer_1.ckpt')
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)

    def ProcessImage(self, imageFileBytes, configs, modelType):
        img = bytes_to_ndarray(imageFileBytes)
        orgImg = img.copy()
        drawImg = img.copy()
        if (img.shape[1] > img.shape[0]):
            formRatio = 480.0 / img.shape[0]
        else:
            formRatio = 480.0 / img.shape[1]
        if (formRatio > 1):
            formRatio = 1
        isImprove = False
        improveBox = [0,0,orgImg.shape[1],orgImg.shape[0]]
        count = 0
        boxes = []
        result = [[]]
        if not isImprove:
            img = cv2.resize(img, (0,0), fx=formRatio, fy=formRatio)
            result = self.ocr.ocr(img=img, cls=False, rec=False)
            for box in result[0]:
                x_min,y_min,x_max,y_max = (quad_coords_to_xyxy(box))
                if x_max - x_min >= 30 and y_max - y_min >= 30:
                    boxes += [box]
        while isImprove:
            count += 1
            print("img: ", img.shape)
            boxes = []
            img = cv2.resize(img, (0,0), fx=formRatio, fy=formRatio)
            # npImg = Image.fromarray(img)
            #detect by paddle
            # if count > 1:
            # result = self.ocr.ocr(img=img, cls=False, rec=False)
            # print("result: ", result)
            # for idx in range(len(result)):
            #     res = result[idx]
            #     for line in res:
            #         print(line)
            # boxes = result[0]
            # print("boxes: ", boxes)
            # print("size: ", len(boxes))

            #improve detection by crop area 
            if count == 1:
                for i in range(4):
                    smallRect = [int(img.shape[1]*0.5)*((i)%2), int(img.shape[0]*0.5)*(i//2), int(img.shape[1]*0.5), int(img.shape[0]*0.5)]
                    smallImg = img[smallRect[1]:smallRect[1]+smallRect[3], smallRect[0]:smallRect[0]+smallRect[2]]
                    result = self.ocr.ocr(img=smallImg, cls=False, rec=False)
                    
                    print("smallRect: ", smallRect)
                    print("small result: ", result)
                    for box in result[0]:
                        for i in range(len(box)):
                            box[i][0] += smallRect[0]
                            box[i][1] += smallRect[1]
                        boxes += [box]
                
                for i in range(2):
                    print("(5^(i%2)) = ", pow(5,i%2))
                    smallRect = [int(img.shape[1]*0.4*((i)%2)), int(img.shape[0]*0.4*((i+1)%2)), int(img.shape[1]/pow(5,i%2)), int(img.shape[0]/pow(5,(i+1)%2))]
                    smallImg = img[smallRect[1]:smallRect[1]+smallRect[3], smallRect[0]:smallRect[0]+smallRect[2]]
                    result = self.ocr.ocr(img=smallImg, cls=False, rec=False)         
                    print("smallRect: ", smallRect)
                    print("small result: ", result)
                    for box in result[0]:
                        for i in range(len(box)):
                            box[i][0] += smallRect[0]
                            box[i][1] += smallRect[1]
                        boxes += [box]

            isImprove = False
            # if len(boxes) == 0:
            #     break
            # coords = []
            # for box in boxes:
            #     coords += box
            # print("coords: ", coords)
            # x_values = [x for x, _ in coords]
            # y_values = [y for _, y in coords]
            # x_min, x_max = min(x_values), max(x_values)
            # y_min, y_max = min(y_values), max(y_values)
            # isImprove = False
            # w = improveBox[2] - improveBox[0]
            # h = improveBox[3] - improveBox[1]
            # if (x_min > img.shape[1]*0.5):
            #     improveBox = [int(w*0.5) + improveBox[0], 0 + improveBox[1], w + improveBox[0], h + improveBox[1]]
            #     isImprove = True
            # elif x_max < img.shape[1]*0.5:
            #     improveBox = [0 + improveBox[0],0 + improveBox[1],int(w*0.5) + improveBox[0],h + improveBox[1]]
            #     isImprove = True

            # w = improveBox[2] - improveBox[0]
            # h = improveBox[3] - improveBox[1]
            # if y_min > img.shape[0]*0.5:
            #     improveBox = [0 + improveBox[0],int(h*0.5) + improveBox[1],w + improveBox[0],h + improveBox[1]]
            #     isImprove = True
            # elif y_max < img.shape[0]*0.5:
            #     improveBox = [0 + improveBox[0],0 + improveBox[1],w + improveBox[0],int(h*0.5) + improveBox[1]]
            #     isImprove = True
            
            # print("improveBox: ", improveBox)
            # if isImprove:
            #     img = orgImg[improveBox[1]:improveBox[3], improveBox[0]:improveBox[2]]

        for i in range(len(boxes)):
            boxes[i] = (quad_coords_to_xyxy(boxes[i]))
            boxes[i] = [boxes[i][0]/formRatio+improveBox[0], boxes[i][1]/formRatio+improveBox[1], boxes[i][2]/formRatio+improveBox[0], boxes[i][3]/formRatio+improveBox[1]]
        
        if len(boxes) == 0:
            boxes += [improveBox]
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
                w = min(orgImg.shape[1], x_max + int(w*externRatio*0.5)) - x
                h = min(orgImg.shape[0], y_max + int(h*externRatio*0.5)) - y
                origBoxes.append([int(x),int(y),int((x + w)),int((y + h))])
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
                origBoxes.append([int(x),int(y),int((x + w)),int((y + h))])
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