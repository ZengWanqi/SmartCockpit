import cv2
import pandas as pd
import numpy as np
import constants
import face_seat_control
import face_feature_extract


class face_recognition:
    def __init__(self):
        self.seat_controller = face_seat_control.FaceSeatControl()  # 座椅控制器实例
        self.feature_extractor = face_feature_extract.FaceFeatureExtractor()  # 人脸特征提取器实例
        self.features_df = pd.read_csv(constants.FEATURES_ALL_CSV_PATH)  # 读取特征数据
        self.camera = cv2.VideoCapture(constants.CAMERA_INDEX)  # 摄像头实例
        self.initialize()

    def initialize(self):
        if not self.seat_controller.connect_to_seat_controller():
            input("按Enter键退出...")
            return

    def match_face_feature(self, face_feature_vector):
        min_distance = float('inf')  # 初始化最小距离为无穷大
        matched_user = None  # 初始化匹配的用户
        for idx, row in self.features_df.iterrows():
            db_vector = np.array(row[6:])  # 假设第0列为用户ID，后面为特征向量
            distance = np.linalg.norm(face_feature_vector - db_vector)
            if distance < min_distance:
                min_distance = distance
                matched_user = row[0]
        return matched_user, min_distance


    def run(self):
        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)  # 设置窗口名称
        print("开始人脸识别，按 'Q' 或 ESC 键退出...")
        print("按 'S' 键保存当前用户的座椅位置...")
        frame_count = 0  # 帧计数器

        while True:
            ret, frame = self.camera.read()  # 读取摄像头帧
            if not ret:
                print("无法从摄像头读取视频帧")
                break
            frame_count += 1  # 计数帧数
            # 每隔 指定帧数 进行一次人脸识别
            if frame_count % constants.RECOGNITION_INTERVAL_FRAMES == 0 and not self.seat_controller.is_seat_adjusting:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图进行人脸检测
                faces = self.feature_extractor.detector(gray)
                if len(faces) > 0:  # 如果检测到人脸
                    face = faces[0]  # 只处理检测到的第一张人脸
                    shape = self.feature_extractor.shape_predictor(gray, face)  # 获取人脸关键点
                    # 计算人脸特征向量
                    face_feature_vector = self.feature_extractor.face_recognition_model.compute_face_descriptor(frame, shape)
                    matched_user, distance = self.match_face_feature(np.array(face_feature_vector))

                    if distance < constants.RECOGNITION_THRESHOLD:
                        print(f"识别到用户: {matched_user}")
                        self.seat_controller.current_user = matched_user
                        # 调整座椅到识别用户的保存位置
                        self.seat_controller.send_user_to_seat_controller()
                    else:
                        print("未识别到匹配的用户")


def main():
    face_recognition()


if __name__ == "__main__":
    main()
