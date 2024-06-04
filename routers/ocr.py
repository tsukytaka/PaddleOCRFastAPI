# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException, UploadFile, status
from models.OCRModel import *
from models.RestfulModel import *
from paddleocr import PaddleOCR
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
import requests
import os


OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "ch")

router = APIRouter(prefix="/ocr", tags=["OCR"])

ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANGUAGE)

@router.post('/predict-by-file', response_model=RestfulModel, summary="read file")
async def predict_by_file(file: UploadFile):
    imageReader: ImageReader = ImageReader()
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".jpeg",".png")):
        # restfulModel.resultcode = 200
        # restfulModel.message = file.filename
        file_data = file.file
        file_bytes = file_data.read()
        output_file_bytes = file_bytes
        # img = bytes_to_ndarray(file_bytes)
        # result = ocr.ocr(img=img, cls=False, rec=False)
        # print("ocr result: ", result)
        # detect by paddle
        output_file_bytes = imageReader.ProcessImage(file_bytes)

        #recognize by parseq

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only upload file .jpg, .jpeg or .png"
        )

    return Response(output_file_bytes, media_type="image/jpg")
