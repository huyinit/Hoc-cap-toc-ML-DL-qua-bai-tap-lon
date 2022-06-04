import asyncio 
import shutil
import uuid
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
from src.dl_code.predict_image import single_predict
import os

logger = logging.getLogger(__name__)
app = FastAPI()

origins = [
    "http://127.0.0.1:5500/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.after_request
# def after_request(response):
#     response.headers.set('Access-Control-Allow-Origin', '*')
#     response.headers.set('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.set('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     return response

@app.on_event("startup")
async def startup() -> None:
    logger.info("startup_start")
@app.get("/health")
async def health(make_error: bool = None) -> Dict[Any, Any]:
    if make_error:
        raise Exception("test problem.")
    return {}


@app.post("/api/v1/predict", tags=["Predict"])
async def api_predict(
    upload_file: UploadFile = File(...)) -> dict:
    print("Go heerrre ")
    file_name = upload_file.filename
    print(file_name)
    # print(os.path.abspath(file_name))
    ext = file_name.split(".")[-1]
    allow_ext = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    if ext not in allow_ext:
        return {
            "status": 500,
            "message": "The file is not in the correct format."
        }
    file_name = "static/image/" + str(uuid.uuid4()) + ".jpg" #UUID là một số nhận dạng duy nhất phổ biến => Universally Unique Identifier
    print(file_name)
    with open(file_name, 'wb') as image_save:# lưu 1 cái ảnh mới từ các định dạng khác về định dạng jpg
        shutil.copyfileobj(upload_file.file, image_save)
        # os.remove("sales_1.txt")

    result = await single_predict(file_name)
    print(result)
    return {
        "status": 200,
        "data": result
    }
