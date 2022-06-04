from typing import Any

from fastapi import File, UploadFile
from tensorflow import keras
import numpy as np
import cv2


def model_load(path_model: str)-> Any:
  keras.backend.clear_session()
  model = keras.models.load_model(path_model)
  return model


def load_image_into_numpy_array(data):
  npimg = np.frombuffer(data, np.uint8)
  image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image


async def single_predict(path_file: str) -> dict:
  model = model_load('model/cifar10.h5')
  image = cv2.imread(path_file)
  image = cv2.resize(image, dsize=(32, 32))
  label_dict = {0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
  predict=model.predict(image.reshape(1, 32, 32, 3))
  return {"path_file": path_file,
          "label": label_dict[np.argmax(predict)]}

