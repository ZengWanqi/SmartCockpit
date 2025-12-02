import cv2
import dlib
import pandas as pd
import constants
from datetime import datetime


class CommonContext:
    """
    公共上下文类，存储共享资源
    """

    def __init__(self):
        self.current_user_id = None  # 当前用户ID
        self.serial_port = None  # 串口端口
        self.serial_obj = None  # 串口对象
        self.is_adjusting_seat = False  # 座椅调整状态标志
        try:
            self.features_df = pd.read_csv(constants.FEATURES_ALL_CSV_PATH)  # 读取特征数据
        except (pd.errors.EmptyDataError,FileNotFoundError):
            self.features_df = pd.DataFrame()
        try:
            self.user_info_df = pd.read_csv(constants.USER_INFO_CSV_PATH)  # 读取用户信息数据
        except (pd.errors.EmptyDataError,FileNotFoundError):
            self.user_info_df = pd.DataFrame()
        self.detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
        self.camera = cv2.VideoCapture(constants.CAMERA_INDEX)  # 摄像头对象
        self.shape_predictor = dlib.shape_predictor(constants.FACE_LANDMARK_68_PREDICTOR_DAT_PATH)  # 68点人脸标志预测模型
        self.face_recognition_model = dlib.face_recognition_model_v1(constants.FACE_RECOGNITION_RESNET_MODEL_V1_DAT_PATH)  # 人脸识别模型


def draw_face_box(image, face_box, color=constants.FACE_BOX_COLOR, thickness=2):
    """
    在图像上绘制人脸头像框
    :param image: 输入的图像
    :param face_box: 人脸框坐标 (x, y, w, h)
    :param color: 框的颜色
    :param thickness: 框的线宽
    """
    x, y, w, h = face_box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def save_face_image(image, face_box, save_folder, user_id):
    """
    保存人脸图像到指定文件夹
    :param user_id:
    :param image: 输入的图像
    :param face_box: 人脸框坐标 (x, y, w, h)
    :param save_folder: 保存文件夹路径
    :return: 无
    """
    x, y, w, h = face_box
    y0, y1 = max(y, 0), max(y + h, 0)
    x0, x1 = max(x, 0), max(x + w, 0)
    face_img = image[y0:y1, x0:x1]
    if face_img.size > 0:
        img_count = len(list(save_folder.glob(f"{user_id}*.jpg"))) + 1 if user_id else len(list(save_folder.glob("*.jpg"))) + 1
        img_path = save_folder / f"{user_id}{img_count}.jpg" if user_id else save_folder / f"{img_count}.jpg"
        cv2.imwrite(str(img_path), face_img)
