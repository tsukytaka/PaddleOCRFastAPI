# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException, UploadFile, status, Form
from models.OCRModel import *
from models.RestfulModel import *
from paddleocr import PaddleOCR
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
import requests
import os
import json

OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "ch")

router = APIRouter(prefix="/ocr", tags=["OCR"])

ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANGUAGE)
imageReader = ImageReader()

@router.post('/predict-by-file', response_model=RestfulModel, summary="read file")
async def predict_by_file(file: UploadFile, infos: str=Form(), model:int=1):
    # imageReader: ImageReader = ImageReader()
    # restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".jpeg",".png")):
        configs = json.loads(infos)
        # restfulModel.resultcode = 200
        # restfulModel.message = file.filename
        file_data = file.file
        file_bytes = file_data.read()
        output_file_bytes = file_bytes
        # img = bytes_to_ndarray(file_bytes)
        # result = ocr.ocr(img=img, cls=False, rec=False)
        # print("ocr result: ", result)
        # detect by paddle
        output_file_bytes = imageReader.ProcessImage(file_bytes, configs, model)

        #recognize by parseq

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only upload file .jpg, .jpeg or .png"
        )

    return Response(output_file_bytes, media_type="image/png")

@router.post('/read-file-with-position', response_model=RestfulModel, summary="read file at some position")
async def read_file_with_position(file: UploadFile, positions: str=Form(), model:int=0): 
    # restfulModel: RestfulModel = RestfulModel()
    
    print("file name: ",file.filename)
    # print("positions: ",positions)
    # test_array = [{"tableNo":1,"rowNo":1,"colNo":1,"content":36,"position":[601,796,79,42]}]
    # positions = json.dumps(test_array)
    items = json.loads(positions)
    print("items: ", items)
    if file.filename.endswith((".jpg", ".jpeg",".png")):
        file_data = file.file
        file_bytes = file_data.read()
        output_file_bytes = file_bytes
        items, output_file_bytes = imageReader.ReadImageWithPos(file_bytes, items, model)
        print("items: ", items)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only upload file .jpg, .jpeg or .png"
        )
    return Response(json.dumps(items))
    # return Response(output_file_bytes, media_type="image/png")