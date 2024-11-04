# server.py
# html 사용
from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import torch
from werkzeug.utils import secure_filename
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

app = Flask(__name__)

UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# 위험도 분석 함수 정의
def analyseImage(thresholdLower, thresholdUpper):
    imageFileList = os.listdir(app.config["UPLOAD_FOLDER"])
    result_data = []

    for image_name in imageFileList:
        im = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], image_name))

        # 탐지할 균열 클래스 설정
        classes = ["diagonal_crack", "horizontal_crack", "vertical_crack"]

        # Detectron2 설정 로드
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        # 학습 모델 설정 경로
        cfg.MODEL.WEIGHTS = os.path.join("output", "model_final.pth")

        # 로드할 클래스 수를 3개로 설정
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresholdLower / 100
        cfg.MODEL.DEVICE = "cpu"

        # 보안 설정: weights_only=True
        checkpoint = torch.load(
            cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"), weights_only=True
        )
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)

        # 최대 신뢰도 점수 계산
        max_score = (
            max(outputs["instances"].scores.tolist()) * 100
            if len(outputs["instances"].scores.tolist()) > 0
            else 0
        )

        # 위험도 판정
        if max_score > thresholdUpper:
            risk_level = "High"
        elif thresholdLower <= max_score <= thresholdUpper:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        result_data.append(
            {"filename": image_name, "max_score": max_score, "risk_level": risk_level}
        )

    return result_data


# Flask 페이지 및 엔드포인트 설정
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    thresholdLower = int(request.form["thresholdLower"])
    thresholdUpper = int(request.form["thresholdUpper"])

    # 업로드된 파일 처리
    if "files[]" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    # 업로드된 모든 파일을 저장
    for file in request.files.getlist("files[]"):
        if file and file.filename.endswith((".png", ".jpg", ".jpeg", ".tiff")):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    # 분석 수행
    result = analyseImage(thresholdLower, thresholdUpper)

    # 결과 반환
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
