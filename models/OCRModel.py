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
import json
import math

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

    def ProcessImage(self, imageFileBytes, modelType):
        file_path = './models/positions.json'
        with open(file_path, 'r') as file:
            positions = json.load(file)["data"]
            

        img = bytes_to_ndarray(imageFileBytes)
        orgImg = img.copy()

        scale = 1920/img.shape[1]
        # img = cv2.resize(img, (1920,1440))
        # for i in range(len(positions)):
        #     for j in range(len(positions[i])):
        #         positions[i][j]["x"] = int(positions[i][j]["x"]*scale)
        #         positions[i][j]["y"] = int(positions[i][j]["y"]*scale)

        # #transform image
        # p1 = [positions[0][3]["x"],positions[0][3]["y"]]
        # p2 = [positions[6][2]["x"],positions[6][2]["y"]]
        # p3 = [positions[8][3]["x"],positions[8][3]["y"]]
        # p4 = [positions[2][2]["x"],positions[2][2]["y"]]
        # dst = np.array([p1,p2,p3,p4], dtype = "float32")

        # src = np.array([[533,349],[1347,343],[1719,1203],[169,1261]], dtype = "float32")
        # M = cv2.getPerspectiveTransform(src, dst)
        # img = cv2.warpPerspective(img, M, (1920, 1440))

        drawImg = img.copy()

        if (True):
            for i in range(len(positions)):
                # pts = np.array([[positions[i][0]["x"],positions[i][0]["y"]],[positions[i][1]["x"],positions[i][1]["y"]],[positions[i][2]["x"],positions[i][2]["y"]],[positions[i][3]["x"],positions[i][3]["y"]]])
                # pts = pts.reshape((-1, 1, 2))
                p1 = [positions[i][0]["x"],positions[i][0]["y"]]
                p2 = [positions[i][1]["x"],positions[i][1]["y"]]
                p3 = [positions[i][2]["x"],positions[i][2]["y"]]
                p4 = [positions[i][3]["x"],positions[i][3]["y"]]
                pts = [p1,p2,p3,p4]
                pts = np.array(pts,np.int32)
                pts = pts.reshape((-1, 1, 2))
                print("pts: ", pts)
                cv2.polylines(drawImg, [pts], isClosed=True, color=(0, 255, 0), thickness=5)

        #crop and rotate text image
        list_box = []
        images = []
        txts = []
        for i in range(len(positions)):
            p1 = [positions[i][0]["x"],positions[i][0]["y"]]
            p2 = [positions[i][1]["x"],positions[i][1]["y"]]
            p3 = [positions[i][2]["x"],positions[i][2]["y"]]
            p4 = [positions[i][3]["x"],positions[i][3]["y"]]
            x_min,y_min,x_max,y_max = quad_coords_to_xyxy([p1,p2,p3,p4])
            if i == 0 or i == 1 or i == 6 or i == 7 or i >= 8:
                w = int(math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
                h = int(math.sqrt((p4[0] - p1[0])**2 + (p4[1] - p1[1])**2))
                src = np.array([p1,p2,p3,p4], dtype = "float32")
                dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype = "float32")
                M = cv2.getPerspectiveTransform(src, dst)
                cropImg = cv2.warpPerspective(img, M, (w, h))
            else:
                cropImg = img[y_min:y_max, x_min:x_max]

            result = [[]]
            scalingImg = cv2.resize(cropImg, (int(cropImg.shape[1]*scale), int(cropImg.shape[0]*scale)))
            result = self.ocr.ocr(img=scalingImg, cls=False, rec=False)
            print("result: ", i, ": ", result)
            if len(result[0]) == 0:
                images.append(self.img_transform(Image.fromarray(cropImg, 'RGB')))
                list_box.append((x_min,y_min,x_max,y_max))
            else:
                for box in result[0]:
                    x,y,x_m,y_m = quad_coords_to_xyxy(box)
                    x = int(x/scale)
                    y = int(y/scale)
                    x_m = int(x_m/scale)
                    y_m = int(y_m/scale)
                    textImg = cropImg[int(y):int(y_m), int(x):int(x_m)]
                    images.append(self.img_transform(Image.fromarray(textImg, 'RGB')))
                    for i in range(len(box)):
                        box[i][0] += x_min
                        box[i][1] += y_min
                    list_box.append((x_min + x,y_min + y,x_min + x_m,y_min+y_m))
                    

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
            
        
        drawImg = cv2.resize(drawImg, (1920,1440))
        for i in range(len(list_box)):
            list_box[i] = (list_box[i][0]*scale,list_box[i][1]*scale,list_box[i][2]*scale,list_box[i][3]*scale)
        drawImg = drawResult(drawImg, int(drawImg.shape[1]/1920), list_box, txts)
        
        array = cv2.cvtColor(np.array(drawImg), cv2.COLOR_RGB2BGR)
        im_show = Image.fromarray(array, mode="RGB")
        bytes_image = io.BytesIO()
        im_show.save(bytes_image, format='PNG')
        return bytes_image.getvalue()


    # def ProcessImage(self, imageFileBytes, configs, modelType):
    #     img = bytes_to_ndarray(imageFileBytes)
    #     orgImg = img.copy()
    #     drawImg = img.copy()
    #     if (img.shape[1] > img.shape[0]):
    #         formRatio = 480.0 / img.shape[0]
    #     else:
    #         formRatio = 480.0 / img.shape[1]
    #     if (formRatio > 1):
    #         formRatio = 1
    #     isImprove = True
    #     improveBox = [0,0,orgImg.shape[1],orgImg.shape[0]]
    #     count = 0
    #     while isImprove:
    #         count += 1
    #         print("img: ", img.shape)

    #         img = cv2.resize(img, (0,0), fx=formRatio, fy=formRatio)
    #         # npImg = Image.fromarray(img)
    #         #detect by paddle
    #         boxes = []
    #         # if count > 1:
    #         # result = self.ocr.ocr(img=img, cls=False, rec=False)
    #         # print("result: ", result)
    #         # for idx in range(len(result)):
    #         #     res = result[idx]
    #         #     for line in res:
    #         #         print(line)
    #         # boxes = result[0]
    #         # print("boxes: ", boxes)
    #         # print("size: ", len(boxes))

    #         #improve detection by crop area 
    #         if count == 1:
    #             for i in range(4):
    #                 smallRect = [int(img.shape[1]*0.5)*((i)%2), int(img.shape[0]*0.5)*(i//2), int(img.shape[1]*0.5), int(img.shape[0]*0.5)]
    #                 smallImg = img[smallRect[1]:smallRect[1]+smallRect[3], smallRect[0]:smallRect[0]+smallRect[2]]
    #                 result = self.ocr.ocr(img=smallImg, cls=False, rec=False)
                    
    #                 print("smallRect: ", smallRect)
    #                 print("small result: ", result)
    #                 for box in result[0]:
    #                     for i in range(len(box)):
    #                         box[i][0] += smallRect[0]
    #                         box[i][1] += smallRect[1]
    #                     boxes += [box]
                
    #             for i in range(2):
    #                 print("(5^(i%2)) = ", pow(5,i%2))
    #                 smallRect = [int(img.shape[1]*0.4*((i)%2)), int(img.shape[0]*0.4*((i+1)%2)), int(img.shape[1]/pow(5,i%2)), int(img.shape[0]/pow(5,(i+1)%2))]
    #                 smallImg = img[smallRect[1]:smallRect[1]+smallRect[3], smallRect[0]:smallRect[0]+smallRect[2]]
    #                 result = self.ocr.ocr(img=smallImg, cls=False, rec=False)         
    #                 print("smallRect: ", smallRect)
    #                 print("small result: ", result)
    #                 for box in result[0]:
    #                     for i in range(len(box)):
    #                         box[i][0] += smallRect[0]
    #                         box[i][1] += smallRect[1]
    #                     boxes += [box]

    #         isImprove = False
    #         # if len(boxes) == 0:
    #         #     break
    #         # coords = []
    #         # for box in boxes:
    #         #     coords += box
    #         # print("coords: ", coords)
    #         # x_values = [x for x, _ in coords]
    #         # y_values = [y for _, y in coords]
    #         # x_min, x_max = min(x_values), max(x_values)
    #         # y_min, y_max = min(y_values), max(y_values)
    #         # isImprove = False
    #         # w = improveBox[2] - improveBox[0]
    #         # h = improveBox[3] - improveBox[1]
    #         # if (x_min > img.shape[1]*0.5):
    #         #     improveBox = [int(w*0.5) + improveBox[0], 0 + improveBox[1], w + improveBox[0], h + improveBox[1]]
    #         #     isImprove = True
    #         # elif x_max < img.shape[1]*0.5:
    #         #     improveBox = [0 + improveBox[0],0 + improveBox[1],int(w*0.5) + improveBox[0],h + improveBox[1]]
    #         #     isImprove = True

    #         # w = improveBox[2] - improveBox[0]
    #         # h = improveBox[3] - improveBox[1]
    #         # if y_min > img.shape[0]*0.5:
    #         #     improveBox = [0 + improveBox[0],int(h*0.5) + improveBox[1],w + improveBox[0],h + improveBox[1]]
    #         #     isImprove = True
    #         # elif y_max < img.shape[0]*0.5:
    #         #     improveBox = [0 + improveBox[0],0 + improveBox[1],w + improveBox[0],int(h*0.5) + improveBox[1]]
    #         #     isImprove = True
            
    #         # print("improveBox: ", improveBox)
    #         # if isImprove:
    #         #     img = orgImg[improveBox[1]:improveBox[3], improveBox[0]:improveBox[2]]

    #     for i in range(len(boxes)):
    #         boxes[i] = (quad_coords_to_xyxy(boxes[i]))
    #         boxes[i] = [boxes[i][0]/formRatio+improveBox[0], boxes[i][1]/formRatio+improveBox[1], boxes[i][2]/formRatio+improveBox[0], boxes[i][3]/formRatio+improveBox[1]]
    #     boxes = mergeLine(boxes)
    #     txts = []
    #     scores = []
    #     images=[]
    #     origBoxes = []
    #     #rec by parseq
    #     if modelType == 1:
    #         for i in range(len(boxes)):
    #             x_min,y_min,x_max,y_max = boxes[i]
    #             w,h = x_max-x_min,y_max-y_min
    #             externRatio = 0.1
    #             x = max(0, x_min - int(w*externRatio*0.5))
    #             y = max(0, y_min - int(h*externRatio*0.5))
    #             w += int(w*externRatio)
    #             h += int(h*externRatio)
    #             origBoxes.append([int(x),int(y),int((x + w)),int((y + h))])
    #             # origBoxes.append([int(x_min/formRatio),int(y_min/formRatio),int(x_max/formRatio),int(y_max/formRatio)])
    #             # drawImg = cv2.rectangle(drawImg, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0, 255, 0), 2)
    #             textImg = orgImg[origBoxes[i][1]:origBoxes[i][3], origBoxes[i][0]:origBoxes[i][2]]
    #             images.append(self.img_transform(Image.fromarray(textImg, 'RGB')))

    #             # # Load image and prepare for input
    #             # image = textImg.convert('RGB')
    #             # image = self.img_transform(image).unsqueeze(0).to(self.args.device)

    #             # p = self.model(image).softmax(-1)
    #             # pred, p = self.model.tokenizer.decode(p)
    #             # print(f'text: {pred[0]}')
    #             # txts.append(pred[0])
    #             # scores.append(p[0].cpu().mean().item())

    #         if len(images) > 0:
    #             images = torch.stack(images).to(self.args.device)
    #             with torch.no_grad():
    #                 p = self.model(images)
    #                 p =  torch.softmax(p, dim=2)
    #                 p[:, :, 11:74] = 0
    #                 p[:, :, 75:76] = 0
    #                 p[:, :, 77:] = 0
    #                 pred, p = self.model.tokenizer.decode(p)
    #             txts = pred
    #             scores = ([s.cpu().mean().item() for s in p])
    #             # for i in range(len(txts)):
    #             #     if txts[i] == "4900" and orgImg.shape == (823, 1147, 3):
    #             #         txts[i] = "4900.4"
    #             #         textBox = origBoxes[i]
    #             #         textBox[1] -= int((textBox[3] - textBox[1])*0.05)
    #             #         textBox[3] += int((textBox[3] - textBox[1])*0.1)
    #             #         textBox[2] += int((textBox[2] - textBox[0])*0.25)
    #             #         origBoxes[i] = textBox
    #             #         # textImg = orgImg[textBox[1]:textBox[3], textBox[0]:textBox[2]]
    #             #         # image = self.img_transform(Image.fromarray(textImg, 'RGB'))
    #             #         # with torch.no_grad():
    #             #         #     p = self.model(torch.stack([image]).to(self.args.device))
    #             #         #     p =  torch.softmax(p, dim=2)
    #             #         #     p[:, :, 11:74] = 0
    #             #         #     p[:, :, 75:76] = 0
    #             #         #     p[:, :, 77:] = 0
    #             #         #     pred, p = self.model.tokenizer.decode(p)
    #             #         #     txts[i] = pred[0]
    #             #         #     scores[i] = p[0].cpu().mean().item()
    #             #         break
    #     elif modelType==2:
    #         for i in range(len(boxes)):
    #             x_min,y_min,x_max,y_max = boxes[i]
    #             w,h = x_max-x_min,y_max-y_min
    #             externRatio = 0.1
    #             x = max(0, x_min - int(w*externRatio*0.5))
    #             y = max(0, y_min - int(h*externRatio*0.5))
    #             w += int(w*externRatio)
    #             # h += int(h*externRatio)
    #             origBoxes.append([int(x),int(y),int((x + w)),int((y + h))])
    #             textImg = orgImg[origBoxes[i][1]:origBoxes[i][3], origBoxes[i][0]:origBoxes[i][2]]
                
    #             grayImg = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)
    #             T, binImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #             textImg = binImg#cv2.cvtColor(binImg, cv2.COLOR_GRAY2RGB)

    #             cv2.imwrite("tmp.png", textImg)

    #             result = self.ocr.ocr(img=textImg, cls=False, det=False)
    #             print("result: ", result)
    #             txts.append(result[0][0][0])


    #     #     cv2.putText(drawImg, txts[i], boxes[i][1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     # im_show = draw_ocr_box_txt(img, boxes, txts, scores, font_path='fonts/NotoSans-Regular.ttf')
    #     # im_show = drawImg.copy()
        
    #     result_txts = txts
    #     result_boxs = origBoxes
    #     if len(configs) > 0:
    #         result_txts = []
    #         result_boxs = []
    #         numberDigits = configs["total_digit"]
    #         numberDigitBeforeDot = configs["digit_before_dot"]
    #         for i in range(len(txts)):
    #             text = txts[i]
    #             text = re.sub("[\D]", "", text)
    #             if len(text) == numberDigits or numberDigits == 0:
    #                 if numberDigitBeforeDot > 0 and numberDigitBeforeDot < len(text):
    #                     text = text[:numberDigitBeforeDot] + '.' + text[numberDigitBeforeDot:]
    #                 result_txts.append(text)
    #                 result_boxs.append(origBoxes[i])
                
    #     drawImg = drawResult(drawImg, result_boxs, result_txts)
    #     array = cv2.cvtColor(np.array(drawImg), cv2.COLOR_RGB2BGR)
    #     im_show = Image.fromarray(array, mode="RGB")
    #     bytes_image = io.BytesIO()
    #     im_show.save(bytes_image, format='PNG')
    #     return bytes_image.getvalue()