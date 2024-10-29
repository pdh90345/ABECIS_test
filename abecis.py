# importing libraries
from operator import truediv
import warnings
from PyQt6 import QtCore, QtWidgets
from tkinter import scrolledtext
from PyQt6.QtWidgets import *
from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import gdown
import sys
import os
import re
import time
import sys
import threading
from datetime import datetime

# For Extracting Metadata
from PIL import Image
from PIL.ExifTags import TAGS

# For quantification
import numpy as np

# Generating Report
from showinfm import show_in_file_manager

# Detectron 2 libraries
import cv2
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger

# Detectron2의 로깅 설정을 초기화 : 로그 메시지를 콘솔에 출력하도록 설정
setup_logger()
# 사용 중지된 기능에 대한 경고 메시지 무시
warnings.filterwarnings("ignore")


# 진행률을 관리하는 작업자 클래스
class PercentageWorker(QtCore.QObject):
    # 시그널 정의: 작업이 시작, 완료되거나 진행률이 변경될 때 방출
    started = QtCore.pyqtSignal()  # 작업 시작 시그널
    finished = QtCore.pyqtSignal()  # 작업 완료 시그널
    percentageChanged = QtCore.pyqtSignal(int)  # 진행률 변경 시그널

    def __init__(self, parent=None):
        super().__init__(parent)
        self._percentage = 0  # 초기 진행률을 0으로 설정

    # 진행률 Getter
    @property
    def percentage(self):
        return self._percentage  # 현재 진행률 반환

    # 진행률 Setter (값이 변경되면 시그널 방출)
    @percentage.setter
    def percentage(self, value):
        if self._percentage == value:
            return
        self._percentage = value  # 진행률 값 변경
        self.percentageChanged.emit(self.percentage)  # 진행률 변경 시그널 방출

    # 작업 시작 메서드
    def start(self):
        self.started.emit()  # 작업 시작 시그널 방출

    # 작업 완료 메서드
    def finish(self):
        self.finished.emit()  # 작업 완료 시그널 방출


# 작업을 수행하지 않는 더미 작업자 클래스
class FakeWorker:
    def start(self):
        pass  # 아무 작업도 하지 않음

    def finish(self):
        pass  # 아무 작업도 하지 않음

    # 진행률 Getter (항상 0 반환)
    @property
    def percentage(self):
        return 0  # 항상 0 반환

    # 진행률 Setter (아무 작업도 하지 않음)
    @percentage.setter
    def percentage(self, value):
        pass  # 아무 작업도 하지 않음


def analyseImage(
    foo,
    dir_path,
    thresholdLower,
    thresholdUpper,
    ResultPossibleFolder,
    ResultConfidentFolder,
    baz="1",
    worker=None,
):
    imageFileList = []
    # 주어진 디렉토리에서 이미지 파일을 가져옵니다.
    folder_dir = dir_path
    for images in os.listdir(folder_dir):
        # 확장자가 png, jpg, jpeg, tiff인 파일만 선택
        if (
            images.endswith(".png")
            or images.endswith(".jpg")
            or images.endswith(".jpeg")
            or images.endswith(".tiff")
        ):
            imageFileList.append(images)
    # 총 이미지 개수를 계산
    total_images = len(imageFileList)
    # 분석할 이미지가 있으면 분석 시작
    if total_images > 0:
        print("Analyzing " + str(total_images) + " Images...")
        # 진행 상태를 추적하기 위한 worker 설정
        if worker is None:
            worker = FakeWorker()
        worker.start()
        current_id = 0
        while worker.percentage < 100:
            if current_id < (total_images):
                # 멀티스레딩으로 이미지 분석 (현재 이미지 선택)
                image_name = imageFileList[current_id]
                worker.percentage += (1 / total_images) * 100
                percent = int(worker.percentage) + 1
                if percent > 100:
                    percent = 100
                print("[" + str(percent) + "%] Analyzing " + str(image_name))  # mask

                # 진행률이 100%일 때 완료 메시지 출력
                if percent == 100:
                    print("\nCrack Analysis Completed. You may generate a report.")

                # 이미지 파일을 읽어들임 (OpenCV 사용)
                im = cv2.imread(os.path.join(folder_dir, image_name))

                # 탐지할 균열 클래스 설정
                classes = ["diagonal_crack", "horizontal_crack", "vertical_crack"]

                # Detectron2 설정 불러오기
                cfg = get_cfg()
                cfg.merge_from_file(
                    model_zoo.get_config_file(
                        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                    )
                )
                # 데이터셋 설정 (훈련 데이터셋, 테스트 데이터셋)
                cfg.DATASETS.TRAIN = ("category_train",)
                cfg.DATASETS.TEST = ()
                cfg.DATALOADER.NUM_WORKERS = 2
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )

                cfg.SOLVER.IMS_PER_BATCH = 2
                cfg.SOLVER.BASE_LR = 0.00025
                cfg.SOLVER.MAX_ITER = 3000
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
                    3  # 클래스 개수를 균열 종류에 맞게 설정
                )

                # 출력 디렉토리 생성
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

                # 사전 학습된 모델의 가중치 파일 로드
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
                cfg.MODEL.DEVICE = "cpu"
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresholdLower / 100
                cfg.DATASETS.TEST = ("crack_test",)

                # 모델을 이용해 이미지 분석
                predictor = DefaultPredictor(cfg)
                # Detect
                outputs = predictor(im)
                # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
                # print(outputs["instances"].pred_classes)
                # print(outputs["instances"].pred_boxes)
                # 결과 출력 및 마스크 그리기
                MetadataCatalog.get("category_train").set(thing_classes=classes)
                microcontroller_metadata = MetadataCatalog.get("category_train")
                # Visualizer로 탐지된 객체 시각화
                v = Visualizer(
                    im[:, :, ::-1], metadata=microcontroller_metadata, scale=1.2
                )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                # 탐지된 객체의 신뢰도 점수 계산
                if len(outputs["instances"].scores.tolist()) > 0:
                    max_score = max(outputs["instances"].scores.tolist()) * 100
                else:
                    max_score = 0
                # 결과 이미지 저장 폴더 설정 (신뢰도에 따라 결정)
                confidence_type = ""
                # 신뢰도가 상한선보다 높으면 "Confident Crack" 폴더에 저장
                if max_score > thresholdUpper:
                    image_output_path = os.path.join(ResultConfidentFolder, image_name)
                    cv2.imwrite(image_output_path, out.get_image()[:, :, ::-1])  # mask
                    confidence_type = "Confident Crack"
                # 신뢰도가 하한선과 상한선 사이면 "Possible Crack" 폴더에 저장
                elif (max_score >= thresholdLower) and (max_score <= thresholdUpper):
                    image_output_path = os.path.join(ResultPossibleFolder, image_name)
                    cv2.imwrite(image_output_path, out.get_image()[:, :, ::-1])  # mask
                    confidence_type = "Possible Crack"
                else:
                    image_output_path = ""
                if image_output_path != "":
                    # 이미지 마스크 저장
                    mask_array = outputs["instances"].pred_masks.to("cpu").numpy()
                    num_instances = mask_array.shape[0]
                    mask_array = np.moveaxis(mask_array, 0, -1)
                    mask_array_instance = []
                    output = np.zeros_like(im)  # black
                    for i in range(num_instances):
                        mask_array_instance.append(mask_array[:, :, i : (i + 1)])
                        output = np.where(mask_array_instance[i] == True, 255, output)
                    image_output_path_modified = image_output_path
                    # 파일 확장자에 맞춰 저장
                    image_output_path_modified = image_output_path_modified.replace(
                        ".png", "_mask.png"
                    )
                    image_output_path_modified = image_output_path_modified.replace(
                        ".jpg", "_mask.jpg"
                    )
                    image_output_path_modified = image_output_path_modified.replace(
                        ".jpeg", "_mask.jpeg"
                    )
                    image_output_path_modified = image_output_path_modified.replace(
                        ".tiff", "_mask.tiff"
                    )
                    # 결과 마스크 이미지를 저장
                    cv2.imwrite(image_output_path_modified, output)
                    # 골격화 연산 적용 (모폴로지 연산을 이용해 균열 길이 추정)

                    # 이미지를 그레이스케일로 읽기
                    img = cv2.imread(image_output_path_modified, 0)

                    # 이미지를 이진화 (Thresholding)
                    ret, img = cv2.threshold(img, 127, 255, 0)

                    # Step 1: 빈 골격 이미지 생성
                    size = np.size(img)
                    skel = np.zeros(img.shape, np.uint8)

                    # Step 2: Cross-shaped 커널 생성
                    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

                    # Step 3~4: 모폴로지 연산을 통한 골격화 반복 수행
                    while True:
                        # Step 2: 이미지를 열기 연산
                        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
                        # Step 3: 원본 이미지에서 열기 연산을 빼서 골격화 부분 추출
                        temp = cv2.subtract(img, open_img)
                        # Step 4: 원본 이미지를 침식시켜 골격을 정제
                        eroded = cv2.erode(img, element)
                        skel = cv2.bitwise_or(skel, temp)
                        img = eroded.copy()
                        # Step 5: 이미지에 더 이상 흰색 픽셀이 없으면 루프 종료
                        if cv2.countNonZero(img) == 0:
                            break

                    image_output_path_skel = image_output_path
                    image_output_path_skel = image_output_path_skel.replace(
                        ".png", "_length_estimation.png"
                    )
                    image_output_path_skel = image_output_path_skel.replace(
                        ".jpg", "_length_estimation.jpg"
                    )
                    image_output_path_skel = image_output_path_skel.replace(
                        ".jpeg", "_length_estimation.jpeg"
                    )
                    image_output_path_skel = image_output_path_skel.replace(
                        ".tiff", "_mask.tiff"
                    )
                    cv2.imwrite(image_output_path_skel, skel)

                    # 메타데이터 가져오기 및 커버리지 분석
                    # 이미지 파일 열기
                    date_taken = ""
                    image = Image.open(os.path.join(folder_dir, image_name))
                    # EXIF 메타데이터 추출
                    exifdata = image.getexif()
                    # EXIF 데이터에서 각 태그 확인
                    for tagid in exifdata:
                        # getting the tag name instead of tag id
                        tagname = TAGS.get(tagid, tagid)
                        # passing the tagid to get its respective value
                        value = exifdata.get(tagid)
                        if tagname == "DateTime":
                            date_taken = str(value)
                    # 탐지된 클래스명 추출
                    predicted_classes = outputs["instances"].pred_classes.tolist()
                    predicted_classes_names = []
                    # 중복된 클래스명 제거 후 정렬
                    for x in predicted_classes:
                        predicted_classes_names.append(classes[x])
                    predicted_classes_names = list(
                        dict.fromkeys(predicted_classes_names)
                    )
                    predicted_classes_names.sort()
                    # 균열 커버리지 계산 (%로 변환)
                    number_of_white_pix = np.sum(output == 255)  # 흰색 픽셀 수 계산
                    number_of_black_pix = np.sum(output == 0)  # 검정색 픽셀 수 계산
                    crack_coverage = round(
                        (
                            (
                                number_of_white_pix
                                / (number_of_white_pix + number_of_black_pix)
                            )
                            * 100
                        ),
                        2,
                    )

                    # 균열 길이 계산 (골격화된 이미지에서 흰색 픽셀 수로 계산)
                    crack_length = np.sum(skel == 255)
                    # 분석 결과를 텍스트 파일로 저장
                    image_output_path_modified = image_output_path_modified.replace(
                        ".png", "_data.txt"
                    )
                    image_output_path_modified = image_output_path_modified.replace(
                        ".jpg", "_data.txt"
                    )
                    image_output_path_modified = image_output_path_modified.replace(
                        ".jpeg", "_data.txt"
                    )
                    image_output_path_modified = image_output_path_modified.replace(
                        ".tiff", "_data.txt"
                    )

                    # 파일에 분석 결과 쓰기
                    f = open(image_output_path_modified, "w")
                    f.write(
                        image_name
                        + ","
                        + str(confidence_type)  # 신뢰도 유형
                        + ","
                        + str(date_taken)  # 촬영 날짜
                        + ',"'
                        + str(predicted_classes_names)  # 탐지된 클래스 명
                        .strip("[]")
                        .replace("_", " ")
                        .title()
                        + '",'
                        + str(int(max_score))  # 신뢰도 점수
                        + ","
                        + str(crack_coverage)  # 균열 커버리지
                        + ","
                        + str(crack_length)  # 균열 길이
                    )
                    f.close()

                # 다음 이미지로 이동
                current_id += 1
                worker.finish()
    else:
        print("\nNo image to analyze in " + dir_path)


class MainWindow(QWidget):
    def __init__(self):
        self.dir_path = ""  # 디렉토리 경로
        self.console_out = ""  # 콘솔 출력 메시지
        self.thresholdUpper = 85  # 신뢰도 상한선 (기본값 85%)
        self.thresholdLower = 60  # 신뢰도 하한선 (기본값 60%)
        super().__init__()  # 부모 클래스 초기화
        # 이미지 처리에 필요한 변수들
        self.imageFileList = []  # 분석할 이미지 리스트
        self.total_images = 0  # 총 이미지 개수
        # 검증 과정에 필요한 변수들
        self.verificationImageFileList = []  # 검증할 이미지 리스트
        self.total_verification_images = 0  # 총 검증 이미지 개수
        self.current_image_id = 0  # 현재 검증 중인 이미지 인덱스
        self.verify_mode = ""  # 검증 모드
        # 메인 클래스의 메서드 호출
        self.initUI()  # 사용자 인터페이스 초기화

    def consoleUpdate(self, message):
        self.console_out = message + "\n"  # 주어진 메시지를 콘솔 출력용 문자열에 추가
        print(self.console_out)
        # GUI 콘솔에 출력할 메시지 형식 조정 (줄바꿈 추가)
        self.console_out = re.sub("(.{50})", "\\1\n", self.console_out, 0, re.DOTALL)
        self.Console.setText("Application Status : " + self.console_out)

    def changeThresholdUpper(self):
        # 상한선이 하한선보다 낮은지 확인
        if (
            self.thresholdUpper < self.thresholdLower
        ):  # 오류 메시지 표시 (상한선이 하한선보다 낮으면 오류 발생)
            QMessageBox.critical(
                self,
                "Threshold Value Error.",
                "The Upper Threshold must be greater than Lower threshold and vice versa.",
            )
            # 상한선을 하한선보다 10% 높게 설정
            self.thresholdUpper = self.thresholdLower + 10
            self.thresholdSliderUpper.setValue(self.thresholdUpper)
            self.labelThresholdUpperValue.setText(
                "(" + str(self.thresholdUpper) + " %)"
            )
            # 콘솔에 상한선 값 업데이트 메시지 출력
            self.consoleUpdate(
                "Confidence Score Upper Threshold set at "
                + str(self.thresholdUpper)
                + " %"
            )
        else:
            # 상한선 값이 올바르면, 슬라이더 값으로 업데이트
            self.thresholdUpper = self.sender().value()
            self.labelThresholdUpperValue.setText(
                "(" + str(self.thresholdUpper) + " %)"
            )
            # 콘솔에 상한선 값 업데이트 메시지 출력
            self.consoleUpdate(
                "Confidence Score Upper Threshold set at "
                + str(self.thresholdUpper)
                + " %"
            )

    def changeThresholdLower(self):
        # 하한선이 상한선보다 높은지 확인
        if (
            self.thresholdUpper < self.thresholdLower
        ):  # 상한선이 하한선보다 높지 않으면 오류 메시지 표시
            QMessageBox.critical(
                self,
                "Threshold Value Error.",
                "The Upper Threshold must be greater than Lower threshold and vice versa.",
            )
            # 하한선을 상한선보다 10% 낮게 수정
            self.thresholdLower = self.thresholdUpper - 10
            self.thresholdSliderLower.setValue(self.thresholdLower)
            self.labelThresholdLowerValue.setText(
                "(" + str(self.thresholdLower) + " %)"
            )
            # 콘솔에 하한선 값 업데이트 메시지 출력
            self.consoleUpdate(
                "Confidence Score Lower Threshold set at "
                + str(self.thresholdLower)
                + " %"
            )
        else:
            # 하한선 값이 올바르면 슬라이더 값으로 업데이트
            self.thresholdLower = self.sender().value()
            self.labelThresholdLowerValue.setText(
                "(" + str(self.thresholdLower) + " %)"
            )
            # 콘솔에 하한선 값 업데이트 메시지 출력
            self.consoleUpdate(
                "Confidence Score Lower Threshold set at "
                + str(self.thresholdLower)
                + " %"
            )

    def CheckVerifyFolder(self):
        # 분석된 이미지 폴더가 있는지 확인
        analyzedFolderPath = os.path.join(
            self.dir_path, "Crack_Analysis"
        )  # 메인 분석 폴더
        ResultPossibleFolder = os.path.join(
            analyzedFolderPath, "Possible"
        )  # 가능한 균열 폴더
        ResultConfidentFolder = os.path.join(
            analyzedFolderPath, "Confident"
        )  # 확실한 균열 폴더

        # 폴더 존재 여부 확인
        ResultFolderExist = os.path.exists(analyzedFolderPath)  # 메인 폴더 확인
        ResultPossibleFolderExists = os.path.exists(
            ResultPossibleFolder
        )  # 가능한 균열 폴더 확인
        ResultConfidentFolderExists = os.path.exists(
            ResultConfidentFolder
        )  # 확실한 균열 폴더 확인
        # 모든 폴더가 존재하면 True 반환, 아니면 False 반환
        if (
            ResultFolderExist
            and ResultPossibleFolderExists
            and ResultConfidentFolderExists
        ):
            return True
        else:
            return False

    # Image Analysis Functions
    def PrevImage(self):
        # 현재 이미지 인덱스가 0보다 크면 이전 이미지로 이동
        if self.current_image_id > 0:
            self.current_image_id -= 1  # 인덱스 감소 (이전 이미지로 이동)
        # 첫 번째 이미지가 아니면 '이전' 버튼을 표시
        if self.current_image_id > 0:
            self.btnPrevImage.show()  # 첫 번째 이미지면 '이전' 버튼을 숨김
        else:
            self.btnPrevImage.hide()
        # 마지막 이미지가 아니면 '다음' 버튼을 표시
        if self.current_image_id < self.total_verification_images - 1:
            self.btnNextImage.show()
        else:
            self.btnNextImage.hide()  # 마지막 이미지면 '다음' 버튼을 숨김
        # 검증할 이미지가 있으면 이미지를 표시
        if self.total_verification_images > 0:
            self.showImage(
                os.path.join(
                    self.dir_path,
                    "Crack_Analysis",
                    self.verify_mode,
                    self.verificationImageFileList[self.current_image_id],
                )
            )
            # 현재 이미지 번호와 전체 이미지 개수 콘솔에 출력
            self.consoleUpdate(
                "Image "
                + str(self.current_image_id + 1)
                + " of "
                + str(self.total_verification_images)
            )

    def NextImage(self):
        # 현재 이미지 인덱스가 마지막 이미지보다 작으면 다음 이미지로 이동
        if self.current_image_id < self.total_verification_images - 1:
            self.current_image_id += 1  # 인덱스 증가 (다음 이미지로 이동)

        # 마지막 이미지가 아니면 '다음' 버튼을 표시
        if self.current_image_id < self.total_verification_images - 1:
            self.btnNextImage.show()
        else:
            self.btnNextImage.hide()  # 마지막 이미지면 '다음' 버튼 숨김
        # 첫 번째 이미지가 아니면 '이전' 버튼을 표시
        if self.current_image_id > 0:
            self.btnPrevImage.show()
        else:
            self.btnPrevImage.hide()  # 첫 번째 이미지면 '이전' 버튼 숨김
        # 검증할 이미지가 있으면 이미지를 표시
        if self.total_verification_images > 0:
            self.showImage(
                os.path.join(
                    self.dir_path,
                    "Crack_Analysis",
                    self.verify_mode,
                    self.verificationImageFileList[self.current_image_id],
                )
            )
            # 현재 이미지 번호와 전체 이미지 개수 콘솔에 출력
            self.consoleUpdate(
                "Image "
                + str(self.current_image_id + 1)  # 현재 이미지 번호 (1부터 시작)
                + " of "
                + str(self.total_verification_images)
            )

    def RemoveImage(self):
        # 삭제 확인 메시지 창 생성
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Confirm removal")
        dlg.setText("Are you sure you want to remove this result?")
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )  # Yes와 No 버튼 설정
        dlg.setIcon(QMessageBox.Icon.Question)
        button = dlg.exec()  # 메시지 창 실행 및 사용자 입력 대기

        # 사용자가 'Yes'를 선택한 경우에만 삭제 작업 수행
        if button == QMessageBox.StandardButton.Yes:
            if (
                len(self.verificationImageFileList) > 0
            ):  # 검증 이미지 리스트에 이미지가 있는 경우
                # 삭제할 이미지의 경로 설정
                image_path = os.path.join(
                    self.dir_path,
                    "Crack_Analysis",
                    self.verify_mode,
                    self.verificationImageFileList[self.current_image_id],
                )

                # 이미지 파일 삭제
                try:
                    os.remove(image_path)
                except:
                    self.consoleUpdate("Error removing file " + image_path)

                # 이미지 마스크 파일 경로 설정 및 삭제
                image_output_path_modified = image_path
                image_output_path_modified = image_output_path_modified.replace(
                    ".png", "_mask.png"
                )
                image_output_path_modified = image_output_path_modified.replace(
                    ".jpg", "_mask.jpg"
                )
                image_output_path_modified = image_output_path_modified.replace(
                    ".jpeg", "_mask.jpeg"
                )
                image_output_path_modified = image_output_path_modified.replace(
                    ".tiff", "_mask.tiff"
                )
                try:
                    os.remove(image_output_path_modified)  # 마스크 이미지 파일 삭제
                except:
                    self.consoleUpdate(
                        "Error removing file" + image_output_path_modified
                    )  # 삭제 실패 시 오류 출력

                # 이미지 분석 결과 텍스트 파일 경로 설정 및 삭제
                image_output_path_modified = image_path
                image_output_path_modified = image_output_path_modified.replace(
                    ".png", "_data.txt"
                )
                image_output_path_modified = image_output_path_modified.replace(
                    ".jpg", "_data.txt"
                )
                image_output_path_modified = image_output_path_modified.replace(
                    ".jpeg", "_data.txt"
                )
                image_output_path_modified = image_output_path_modified.replace(
                    ".tiff", "_data.txt"
                )
                try:
                    os.remove(image_output_path_modified)  # 텍스트 파일 삭제
                except:
                    self.consoleUpdate(
                        "Error removing file" + image_output_path_modified
                    )  # 삭제 실패 시 오류 출력

                # 검증 이미지 리스트에서 이미지 제거
                try:
                    self.verificationImageFileList.remove(
                        self.verificationImageFileList[self.current_image_id]
                    )
                    # print(self.verificationImageFileList)
                except:
                    self.consoleUpdate("Error processing removal.")
                # 총 검증 이미지 수 업데이트
                self.total_verification_images = len(self.verificationImageFileList)

                # 남은 이미지가 없으면 관련 버튼 및 이미지 숨기기
                if self.total_verification_images == 0:
                    self.btnPrevImage.hide()
                    self.btnRemoveImage.hide()
                    self.btnZoomImage.hide()
                    self.btnNextImage.hide()
                    self.imageHolder.hide()
                self.consoleUpdate("Image Removed from Results")

                # 이미지 삭제 후 현재 인덱스가 범위를 벗어나면 0으로 설정
                if self.current_image_id > self.total_verification_images - 1:
                    self.current_image_id = 0
                else:
                    self.NextImage()  # 다음 이미지로 이동
        else:
            return  # 사용자가 'No'를 선택한 경우 삭제 작업을 중단

    def ZoomImage(self):
        try:
            # 현재 경로에서 이미지 파일을 읽어옴 (OpenCV 사용)
            image = cv2.imread(
                os.path.join(
                    self.dir_path,  # 디렉토리 경로
                    "Crack_Analysis",  # 분석된 이미지 폴더
                    self.verify_mode,  # 검증 모드에 맞는 폴더 선택
                    self.verificationImageFileList[
                        self.current_image_id
                    ],  # 현재 이미지 파일명
                )
            )
            # OpenCV 창을 통해 이미지를 표시
            cv2.imshow(
                self.verify_mode + " Image. Press any key to close image.", image
            )
            # 사용자가 키를 누를 때까지 창을 유지
            cv2.waitKey(0)
            cv2.destroyAllWindows()  # 창 닫기
        except:
            # 이미지 열기에 실패한 경우 오류 메시지 출력
            QMessageBox.critical(
                self, "Error Opening Image.", "There was an error opening the image."
            )

    def showImage(self, image_path):
        try:
            # 메모리 할당량 수정 (이미지 크기와 관련된 설정)
            os.environ["QT_IMAGEIO_MAXALLOC"] = str(os.path.getsize(image_path))

            # 이미지 파일을 QPixmap으로 불러옴
            self.pixmap = QPixmap(image_path)
            # 이미지 크기를 너비 480 픽셀에 맞추어 조정
            self.pixmap = self.pixmap.scaledToWidth(480)
            # 이미지 홀더에 이미지를 설정하고 화면에 표시
            self.imageHolder.setPixmap(self.pixmap)
            self.imageHolder.show()  # 이미지를 GUI에서 보여줌
        except:
            # 이미지 열기에 실패한 경우 오류 메시지 출력
            QMessageBox.critical(
                self, "Error Opening Image.", "There was an error opening the image."
            )

    def VerifyPossible(self):
        # 현재 이미지 인덱스를 0으로 설정 (첫 번째 이미지로 시작)
        self.current_image_id = 0
        # 콘솔에 "Verifying Possible Results" 메시지 출력
        self.consoleUpdate("Verifying Possible Results")
        self.verify_mode = "Possible"  # 검증 모드를 "Possible"로 설정

        # 폴더 경로가 설정되지 않은 경우 오류 메시지 표시
        if self.dir_path == "":
            # 경고 메시지 출력 (폴더가 선택되지 않음)
            QMessageBox.critical(
                self,
                "No folder selected.",
                "Please select a folder containing the images of analysis.",
            )
        else:
            # 분석된 이미지 폴더가 있는지 확인
            if self.CheckVerifyFolder():
                # 검증할 이미지 리스트 초기화
                self.verificationImageFileList = []
                folder_dir = os.path.join(self.dir_path, "Crack_Analysis", "Possible")

                # Possible 폴더에서 이미지 파일 검색
                for images in os.listdir(folder_dir):
                    # 이미지 파일 형식 확인
                    if (
                        images.endswith(".png")
                        or images.endswith(".jpg")
                        or images.endswith(".jpeg")
                        or images.endswith(".tiff")
                    ):
                        # "mask"나 "length_estimation"이 포함된 파일은 제외
                        if "mask" not in images and "length_estimation" not in images:
                            self.verificationImageFileList.append(images)
                # 총 검증 이미지 수 업데이트
                self.total_verification_images = len(self.verificationImageFileList)
                # 콘솔에 검증할 이미지 개수 출력
                self.consoleUpdate(
                    str(self.total_verification_images)
                    + " images found for verification."
                )

                # 검증할 이미지가 있는 경우 첫 번째 이미지 표시 및 관련 버튼 활성화
                if self.total_verification_images > 0:
                    self.showImage(
                        os.path.join(
                            self.dir_path,
                            "Crack_Analysis",
                            "Possible",
                            self.verificationImageFileList[0],
                        )
                    )
                    # 삭제, 확대, 다음 이미지 버튼 활성화
                    self.btnRemoveImage.show()
                    self.btnZoomImage.show()
                    self.btnNextImage.show()
                else:
                    # 검증할 이미지가 없는 경우 오류 메시지 표시
                    QMessageBox.critical(
                        self,
                        "No Analyzed Results.",
                        "The analyzed folders contain no Possible images.",
                    )
            else:
                # 분석된 결과가 없는 경우 오류 메시지 표시
                QMessageBox.critical(
                    self,
                    "No Analyzed Results.",
                    "There are no analyzed results in this folder. Try running Crack Analysis.",
                )

    def VerifyConfident(self):
        # 현재 이미지 인덱스를 0으로 설정 (첫 번째 이미지로 시작)
        self.current_image_id = 0
        # 콘솔에 "Verifying Confident Results" 메시지 출력
        self.consoleUpdate("Verifying Confident Results")
        self.verify_mode = "Confident"  # 검증 모드를 "Confident"로 설정

        # 폴더 경로가 설정되지 않은 경우 오류 메시지 표시
        if self.dir_path == "":
            # 경고 메시지 출력 (폴더가 선택되지 않음)
            QMessageBox.critical(
                self,
                "No folder selected.",
                "Please select a folder containing the images of analysis.",
            )
        else:
            # 분석된 이미지 폴더가 있는지 확인
            if self.CheckVerifyFolder():
                # 검증할 이미지 리스트 초기화
                self.verificationImageFileList = []
                folder_dir = os.path.join(self.dir_path, "Crack_Analysis", "Confident")
                # Confident 폴더에서 이미지 파일 검색
                for images in os.listdir(folder_dir):
                    # 이미지 파일 형식 확인
                    if (
                        images.endswith(".png")
                        or images.endswith(".jpg")
                        or images.endswith(".jpeg")
                        or images.endswith(".tiff")
                    ):
                        # "mask"나 "length_estimation"이 포함된 파일은 제외
                        if "mask" not in images and "length_estimation" not in images:
                            self.verificationImageFileList.append(images)
                # 총 검증 이미지 수 업데이트
                self.total_verification_images = len(self.verificationImageFileList)
                # 콘솔에 검증할 이미지 개수 출력
                self.consoleUpdate(
                    str(self.total_verification_images)
                    + " images found for verification."
                )

                # 검증할 이미지가 있는 경우 첫 번째 이미지 표시 및 관련 버튼 활성화
                if self.total_verification_images > 0:
                    self.showImage(
                        os.path.join(
                            self.dir_path,
                            "Crack_Analysis",
                            "Confident",
                            self.verificationImageFileList[0],
                        )
                    )
                    # 삭제, 확대, 다음 이미지 버튼 활성화
                    self.btnRemoveImage.show()
                    self.btnZoomImage.show()
                    self.btnNextImage.show()
                else:
                    # 검증할 이미지가 없는 경우 오류 메시지 표시
                    QMessageBox.critical(
                        self,
                        "No Analyzed Results.",
                        "The analyzed folders contain no Confident images.",
                    )
            else:
                # 분석된 결과가 없는 경우 오류 메시지 표시
                QMessageBox.critical(
                    self,
                    "No Analyzed Results.",
                    "There are no analyzed results in this folder. Try running Crack Analysis.",
                )

    def generateReport(self):
        try:
            text_file_paths = []
            # 결과 폴더 경로 생성
            analyzedFolderPath = os.path.join(
                self.dir_path, "Crack_Analysis"
            )  # 분석 폴더 경로
            ResultPossibleFolder = os.path.join(
                analyzedFolderPath, "Possible"
            )  # Possible 폴더 경로
            ResultConfidentFolder = os.path.join(
                analyzedFolderPath, "Confident"
            )  # Confident 폴더 경로
            # 결과 폴더가 존재하는지 확인
            ResultPossibleFolderExists = os.path.exists(
                ResultPossibleFolder
            )  # Possible 폴더 존재 확인
            ResultConfidentFolderExists = os.path.exists(
                ResultConfidentFolder
            )  # Confident 폴더 존재 확인
            # 분석된 텍스트 파일 가져오기
            if ResultPossibleFolderExists:
                for result_text in os.listdir(ResultPossibleFolder):
                    # .txt 파일만 처리
                    if result_text.endswith(".txt"):
                        text_file_paths.append(
                            os.path.join(ResultPossibleFolder, result_text)
                        )
            if ResultConfidentFolderExists:
                for result_text in os.listdir(ResultConfidentFolder):
                    # .txt 파일만 처리
                    if result_text.endswith(".txt"):
                        text_file_paths.append(
                            os.path.join(ResultConfidentFolder, result_text)
                        )
            # 콘솔에 보고서 생성 메시지 출력
            self.consoleUpdate("Generating Report")
            # 자동으로 보고서 파일 이름 생성 (현재 시간 기반)
            now = datetime.now()
            date_time = now.strftime("%b_%d_%Y-%H_%M_%S")  # 보고서 생성 시간
            Report_Filename = (
                "Crack_Analysis_Report_" + date_time + ".csv"
            )  # 보고서 파일 이름
            # 보고서 파일 열기 (쓰기 모드)
            f = open(os.path.join(self.dir_path, Report_Filename), "w")
            output_data = ""
            # 각 텍스트 파일에서 데이터를 읽어와 output_data에 추가
            for data in text_file_paths:
                data_file = open(data, "r")
                output_data += str(data_file.read()) + "\n"
                data_file.close()

            # CSV 형식으로 데이터 작성
            f.write(
                "Filename"  # 파일명
                + ","
                + "Confidence Type"  # 신뢰도 유형
                + ","
                + "Date/Time Taken"  # 촬영 날짜/시간
                + ","
                + "Crack Types"  # 균열 종류
                + ","
                + "Maximum Confidence Score"  # 최대 신뢰도 점수
                + ","
                + "Crack Coverage %"  # 균열 커버리지 비율
                + ","
                + "Total Crack Length (pixels)"  # 총 균열 길이(픽셀 단위)
                + "\n"
                + output_data  # 텍스트 파일에서 읽어온 데이터 추가
                + "\nTotal Files:"  # 파일 총 개수 출력
                + str(len(text_file_paths))
            )
            # 파일 저장 후 닫기
            f.close()
            # 생성된 보고서가 포함된 폴더 열기
            show_in_file_manager(os.path.join(self.dir_path, Report_Filename))
            self.consoleUpdate(
                "Report Successfully Generated"
            )  # 보고서 생성 성공 메시지 출력

        # 예외 처리: 보고서 생성 중 오류가 발생하면 메시지 출력
        except Exception as e:
            QMessageBox.critical(
                self, "Error Generating Report.", "Something went wrong." + str(e)
            )

    def openFolder(self):
        # 폴더 선택 창을 열어 사용자가 폴더를 선택하도록 함
        self.dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory", "")
        # 사용자가 폴더를 선택하지 않았을 때 경고 메시지 표시
        if self.dir_path == "":
            # Throw warning when no folder is selected
            QMessageBox.critical(
                self,
                "No folder selected.",
                "Please select a folder containing the images for crack analysis.",
            )
        else:
            # 선택된 폴더 경로를 labelHelp에 표시
            self.labelHelp.setText("Folder : " + str(self.dir_path))
            # 콘솔에 선택된 폴더 경로 출력
            self.consoleUpdate("Folder Selected at " + str(self.dir_path))

            # 이전 분석 결과와 관련된 버튼 및 이미지 숨기기
            self.btnPrevImage.hide()
            self.btnRemoveImage.hide()
            self.btnZoomImage.hide()
            self.btnNextImage.hide()
            self.imageHolder.hide()

    def downloadPretrainedModel(self):
        # 사전 학습된 모델 경로 설정
        modelPath = os.path.join(os.getcwd(), "output", "model_final.pth")
        # 모델 파일이 존재하는지 확인
        modelExist = os.path.exists(modelPath)

        # 콘솔에 모델 존재 여부 메시지 출력
        self.consoleUpdate("Checking if pretrained model exists.")

        # 모델이 존재하지 않으면 다운로드 진행
        if not modelExist:
            # 모델을 저장할 디렉토리 생성 (필요할 경우)
            os.makedirs(os.path.join(os.getcwd(), "output"))
            # 콘솔에 모델이 없다는 메시지 출력
            self.consoleUpdate("Pretrained Model does not exist. Downloading model.")
            # Google Drive에서 모델 파일 다운로드
            url = "https://drive.google.com/uc?id=1V5biplCaJYHTxIp8achw0KKpm52qPLIa"
            gdown.download(url, modelPath, quiet=False)
            # 콘솔에 모델 다운로드 성공 메시지 출력
            self.consoleUpdate("Model successfully downloaded.")
        else:
            # 모델이 이미 존재하는 경우
            self.consoleUpdate("Pretrained Model exists.")

    def runAnalysis(self):
        # 균열 탐지 알고리즘 실행
        if self.dir_path == "":
            # 폴더가 선택되지 않은 경우 경고 메시지 출력
            QMessageBox.critical(
                self,
                "No folder selected.",
                "Please select a folder containing the images for crack analysis.",
            )
        else:
            # 분석 시작 안내 메시지 출력
            QMessageBox.information(
                self,
                "Running Analysis.",
                "Crack Analysis will begin and will take some time. Plese don't close any windows. Look at the terminal for progress.",
            )
            # 선택한 폴더에서 이미지 파일 불러오기
            self.imageFileList = []  # 이미지 리스트 초기화
            folder_dir = self.dir_path  # 폴더 경로
            for images in os.listdir(folder_dir):
                # 이미지 파일 형식 확인
                if (
                    images.endswith(".png")
                    or images.endswith(".jpg")
                    or images.endswith(".jpeg")
                    or images.endswith(".tiff")
                ):
                    self.imageFileList.append(images)  # 이미지 리스트에 추가

            # 폴더 내 이미지 개수 출력
            self.total_images = len(self.imageFileList)  # 총 이미지 수 저장
            self.consoleUpdate(
                str(self.total_images)
                + " images found in the folder "  # 이미지 개수 출력
                + str(folder_dir)
            )
            # 결과 저장 폴더 생성
            analyzedFolderPath = os.path.join(
                folder_dir, "Crack_Analysis"
            )  # 분석 폴더 경로
            ResultPossibleFolder = os.path.join(
                analyzedFolderPath, "Possible"
            )  # Possible 폴더 경로
            ResultConfidentFolder = os.path.join(
                analyzedFolderPath, "Confident"
            )  # Confident 폴더 경로
            # 폴더 존재 여부 확인
            ResultFolderExist = os.path.exists(analyzedFolderPath)
            ResultPossibleFolderExists = os.path.exists(ResultPossibleFolder)
            ResultConfidentFolderExists = os.path.exists(ResultConfidentFolder)
            # 폴더가 존재하지 않으면 생성
            if not ResultFolderExist:
                os.makedirs(analyzedFolderPath)  # 분석 폴더 생성
                self.consoleUpdate("Created folder " + str(analyzedFolderPath))
            if not ResultPossibleFolderExists:
                os.makedirs(ResultPossibleFolder)  # Possible 폴더 생성
                self.consoleUpdate("Created folder " + str(ResultPossibleFolder))
            if not ResultConfidentFolderExists:
                os.makedirs(ResultConfidentFolder)  # Confident 폴더 생성
                self.consoleUpdate("Created folder " + str(ResultConfidentFolder))

            # 진행률을 추적할 worker 생성
            worker = PercentageWorker()
            worker.percentageChanged.connect(
                self.progress.setValue
            )  # 진행률 변화 시 progress bar 업데이트

            # 분석을 멀티스레드로 실행
            threading.Thread(
                target=analyseImage,  # 분석 함수 호출
                args=(
                    "foo",  # 분석에 필요한 인자 (미사용)
                    self.dir_path,  # 이미지 폴더 경로
                    self.thresholdLower,  # 하한 신뢰도 값
                    self.thresholdUpper,  # 상한 신뢰도 값
                    ResultPossibleFolder,  # Possible 폴더 경로
                    ResultConfidentFolder,  # Confident 폴더 경로
                ),
                kwargs=dict(
                    baz="baz", worker=worker
                ),  # worker 객체를 전달해 진행률 추적
                daemon=True,  # 데몬 스레드로 실행 (프로그램 종료 시 자동 종료)
            ).start()  # 스레드 시작

    def initUI(self):
        # UI 및 메서드 연결 설정
        self.resize(500, 800)  # 창 크기 설정
        self.setWindowTitle(
            "ABECIS v.1.0 - S.M.A.R.T. Construction Research Group"
        )  # 창 제목 설정

        # 라벨 설정 (사용자 안내)
        self.labelHelp = QLabel(
            "To begin, select a folder containing the images of building.", self
        )
        self.labelThresholdUpper = QLabel(
            "3. Set Upper Confidence Score Threshold : ", self
        )
        self.labelThresholdLower = QLabel(
            "2. Set Lower Confidence Score Threshold : ", self
        )

        # 상한/하한 신뢰도 값 라벨
        self.labelThresholdUpperValue = QLabel(
            "(" + str(self.thresholdUpper) + " %)", self  # 현재 상한 신뢰도 값 표시
        )
        self.labelThresholdLowerValue = QLabel(
            "(" + str(self.thresholdLower) + " %)", self  # 현재 하한 신뢰도 값 표시
        )

        # 버튼 설정
        self.btnSelectFolder = QPushButton(
            "1. Select Image Folder", self
        )  # 이미지 폴더 선택 버튼
        self.btnRunAnalysis = QPushButton(
            "3. Run Crack Analysis", self
        )  # 균열 분석 시작 버튼
        self.btnVerifyConfidentResults = QPushButton(
            "4. Verify Confident Results (Optional)", self  # 확실한 결과 검증 버튼
        )
        self.btnVerifyPossibleResults = QPushButton(
            "5. Verify Possible Results (Optional)", self  # 가능성 있는 결과 검증 버튼
        )
        self.btnOpenResults = QPushButton(
            "6. Generate Report and Open Results Folder",
            self,  # 보고서 생성 및 폴더 열기 버튼
        )

        # 상한 신뢰도 슬라이더 설정
        self.thresholdSliderUpper = QSlider(Qt.Orientation.Horizontal, self)
        self.thresholdSliderUpper.setMinimum(0)  # 최소값 0
        self.thresholdSliderUpper.setValue(self.thresholdUpper)  # 현재 값 설정
        self.thresholdSliderUpper.setMaximum(100)  # 최대값 100
        self.thresholdSliderUpper.setTickPosition(
            QSlider.TickPosition.TicksBelow
        )  # 하단에 눈금 표시
        self.thresholdSliderUpper.setTickInterval(10)  # 눈금 간격 설정

        # 하한 신뢰도 슬라이더 설정
        self.thresholdSliderLower = QSlider(Qt.Orientation.Horizontal, self)
        self.thresholdSliderLower.setMinimum(0)
        self.thresholdSliderLower.setValue(self.thresholdLower)
        self.thresholdSliderLower.setMaximum(100)
        self.thresholdSliderLower.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.thresholdSliderLower.setTickInterval(10)

        # 콘솔 출력 라벨 설정 (상태 메시지 표시)
        self.Console = QLabel("", self)
        self.Console.setText(self.console_out)

        # 레이아웃 설정 (위젯 배치)
        vbox = QVBoxLayout()
        vbox.addWidget(self.labelHelp)
        vbox.addWidget(self.btnSelectFolder)

        # 하한 신뢰도 슬라이더를 레이아웃에 추가
        thresholdHboxLower = QHBoxLayout()
        thresholdHboxLower.addWidget(self.labelThresholdLowerValue)
        thresholdHboxLower.addWidget(self.thresholdSliderLower)
        thresholdHboxLower.addWidget(self.labelThresholdLowerValue)

        # 상한 신뢰도 슬라이더를 레이아웃에 추가
        thresholdHboxUpper = QHBoxLayout()
        thresholdHboxUpper.addWidget(self.labelThresholdUpperValue)
        thresholdHboxUpper.addWidget(self.thresholdSliderUpper)
        thresholdHboxUpper.addWidget(self.labelThresholdUpperValue)

        # 이미지 표시 영역 (imageHolder) 설정
        self.imageHolder = QLabel(self)

        # 진행 바 설정
        self.progress = QtWidgets.QProgressBar()

        vbox.addWidget(self.labelThresholdLower)  # 하한 신뢰도 라벨 추가
        vbox.addLayout(thresholdHboxLower)  # 하한 신뢰도 슬라이더 추가
        vbox.addWidget(self.labelThresholdUpper)  # 상한 신뢰도 라벨 추가
        vbox.addLayout(thresholdHboxUpper)  # 상한 신뢰도 슬라이더 추가

        vbox.addWidget(self.btnRunAnalysis)  # 분석 시작 버튼 추가
        vbox.addWidget(self.progress)  # 진행 바 추가

        vbox.addWidget(self.btnVerifyConfidentResults)  # 확실한 결과 검증 버튼 추가
        vbox.addWidget(self.btnVerifyPossibleResults)  # 가능성 있는 결과 검증 버튼 추가
        vbox.addWidget(self.btnOpenResults)  # 보고서 생성 및 폴더 열기 버튼 추가

        # 이미지 분석 도구 버튼 설정 (이전, 삭제, 확대, 다음 버튼)
        self.btnPrevImage = QPushButton("Previous", self)
        self.btnRemoveImage = QPushButton("Remove", self)
        self.btnZoomImage = QPushButton("Zoom", self)
        self.btnNextImage = QPushButton("Next", self)
        analysisHboxUpper = QHBoxLayout()
        analysisHboxUpper.addWidget(self.btnPrevImage)
        analysisHboxUpper.addWidget(self.btnRemoveImage)
        analysisHboxUpper.addWidget(self.btnZoomImage)
        analysisHboxUpper.addWidget(self.btnNextImage)

        vbox.addLayout(analysisHboxUpper)  # 이미지 분석 도구 버튼 추가
        vbox.addWidget(self.imageHolder)  # 이미지 출력 영역 추가

        vbox.addStretch()
        vbox.addWidget(self.Console)  # 콘솔 출력 영역 추가

        # 버튼 및 슬라이더와 함수 연결
        self.thresholdSliderUpper.valueChanged.connect(
            self.changeThresholdUpper
        )  # 상한 신뢰도 변경
        self.thresholdSliderLower.valueChanged.connect(
            self.changeThresholdLower
        )  # 하한 신뢰도 변경
        self.btnSelectFolder.clicked.connect(self.openFolder)  # 폴더 선택 버튼 클릭 시
        self.btnRunAnalysis.clicked.connect(self.runAnalysis)  # 분석 시작 버튼 클릭 시
        self.btnVerifyPossibleResults.clicked.connect(
            self.VerifyPossible
        )  # 가능성 있는 결과 검증
        self.btnVerifyConfidentResults.clicked.connect(
            self.VerifyConfident
        )  # 확실한 결과 검증
        self.btnOpenResults.clicked.connect(
            self.generateReport
        )  # 보고서 생성 버튼 클릭 시

        # 이미지 분석 도구 버튼 연결
        self.btnPrevImage.clicked.connect(self.PrevImage)
        self.btnRemoveImage.clicked.connect(self.RemoveImage)
        self.btnZoomImage.clicked.connect(self.ZoomImage)
        self.btnNextImage.clicked.connect(self.NextImage)

        # 분석 전에 숨겨야 하는 버튼들 숨기기
        self.btnPrevImage.hide()  # '이전' 버튼 숨김
        self.btnRemoveImage.hide()  # '삭제' 버튼 숨김
        self.btnZoomImage.hide()  # '확대' 버튼 숨김
        self.btnNextImage.hide()  # '다음' 버튼 숨김
        self.imageHolder.hide()  # 이미지 표시 영역 숨김

        # 레이아웃 설정 및 창 표시
        self.setLayout(vbox)
        self.show()

        # 사전 학습된 모델 다운로드
        self.downloadPretrainedModel()


if __name__ == "__main__":
    qApp = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(qApp.exec())
