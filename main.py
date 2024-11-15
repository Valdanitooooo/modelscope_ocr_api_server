import os

import uvicorn
from fastapi import FastAPI, UploadFile, File
import math

import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from dotenv import load_dotenv

load_dotenv()

OCR_DETECTION_MODEL_PATH = os.environ.get("OCR_DETECTION_MODEL_PATH",
                                          "iic/cv_resnet18_ocr-detection-db-line-level_damo")
OCR_RECOGNITION_MODEL_PATH = os.environ.get("OCR_RECOGNITION_MODEL_PATH",
                                            "iic/cv_convnextTiny_ocr-recognition-general_damo")

ocr_detection = pipeline(Tasks.ocr_detection, model=OCR_DETECTION_MODEL_PATH)
ocr_recognition = pipeline(Tasks.ocr_recognition, model=OCR_RECOGNITION_MODEL_PATH)


# scripts for crop images
def crop_image(img, position):
    def distance(x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    position = position.tolist()
    for i in range(4):
        for j in range(i + 1, 4):
            if (position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4, 2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1 + x4) / 2, (y1 + y4) / 2, (x2 + x3) / 2, (y2 + y3) / 2)
    img_height = distance((x1 + x2) / 2, (y1 + y2) / 2, (x4 + x3) / 2, (y4 + y3) / 2)

    corners_trans = np.zeros((4, 2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst


def order_point(coor):
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points


app = FastAPI()


@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image_full = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    det_result = ocr_detection(image_full)
    det_result = det_result['polygons']
    data_list = []
    for i in range(det_result.shape[0]):
        pts = order_point(det_result[i])
        image_crop = crop_image(image_full, pts)
        result = ocr_recognition(image_crop)
        print("box: %s" % ','.join([str(e) for e in list(pts.reshape(-1))]))
        print("text: %s" % result['text'])
        data_list += result['text']
    data = "\n".join(data_list)
    return {"data": data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("APP_PORT", 7700)))
