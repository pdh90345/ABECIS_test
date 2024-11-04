# app.py
# api 사용
from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import torch

app = Flask(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# 분석 함수 정의
def analyseImage(image_folder, thresholdLower, thresholdUpper):
    imageFileList = os.listdir(image_folder)
    result_data = []

    for image_name in imageFileList:
        # 이미지 로드
        im = cv2.imread(os.path.join(image_folder, image_name))

        # 탐지할 균열 클래스 설정
        classes = ["diagonal_crack", "horizontal_crack", "vertical_crack"]

        # Detectron2 설정 불러오기
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = os.path.join("output", "model_final.pth")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresholdLower / 100
        cfg.MODEL.DEVICE = "cpu"

        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)

        max_score = (
            max(outputs["instances"].scores.tolist()) * 100
            if len(outputs["instances"].scores.tolist()) > 0
            else 0
        )

        # 위험도 계산을 위한 마스크 생성
        if len(outputs["instances"]) > 0:
            mask = outputs["instances"].pred_masks[0].cpu().numpy()
            output = np.zeros(mask.shape, dtype=np.uint8)  # mask와 동일한 2D 배열 생성
            output = np.where(mask, 255, output)
            crack_coverage = np.sum(output == 255) / output.size * 100
        else:
            crack_coverage = 0

        # 위험도 점수 계산
        confidence_risk = max_score / 100 * 3
        coverage_risk = crack_coverage / 100 * 3
        length_risk = (np.sum(output == 255) / 1000) * 3

        total_risk_score = (
            (confidence_risk * 0.5) + (coverage_risk * 0.3) + (length_risk * 0.2)
        )

        # 위험도 등급 결정
        if total_risk_score >= 2.5:
            risk_level = "상"
        elif 1.5 <= total_risk_score < 2.5:
            risk_level = "중"
        else:
            risk_level = "하"

        result_data.append(
            {
                "filename": image_name,
                "max_score": max_score,
                "crack_coverage": crack_coverage,
                "total_risk_score": total_risk_score,
                "risk_level": risk_level,
            }
        )

    return result_data


# 분석 API 엔드포인트
@app.route("/analyse", methods=["POST"])
def analyse():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    thresholdLower = int(request.form.get("thresholdLower", 50))
    thresholdUpper = int(request.form.get("thresholdUpper", 80))

    for file in request.files.getlist("file"):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    result = analyseImage(app.config["UPLOAD_FOLDER"], thresholdLower, thresholdUpper)

    # 분석이 끝난 후 업로드된 파일 삭제
    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.remove(file_path)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
