import cv2
import numpy as np
import constants
import utilities
import face_seat_control
import face_feature_extract


class FaceRecognition:
    def __init__(self, common_context: utilities.CommonContext, seat_controller: face_seat_control.FaceSeatControl,
                 feature_extractor: face_feature_extract.FaceFeatureExtractor):
        self.current_user_id = common_context.current_user_id  # 当前用户ID
        self.matched_user = None  # 当前匹配的用户ID
        self.recognition_confidence = 0.0  # 识别置信度
        self.seat_controller = seat_controller  # 座椅控制器实例
        self.feature_extractor = feature_extractor  # 人脸特征提取器实例
        self.features_df = common_context.features_df  # 特征数据表
        self.camera = common_context.camera  # 摄像头实例
        if self.seat_controller.serial_obj is None:  # 如果串口未连接
            if not self.seat_controller.connect_to_seat_controller():  # 尝试连接串口
                input("串口连接失败，按回车键退出程序...")
                # exit(1)

    def match_face_feature(self, face_feature_vector):
        min_distance = float('inf')  # 初始化最小距离为无穷大
        matched_user = None  # 初始化匹配的用户
        for idx, row in self.features_df.iterrows():
            db_vector = np.array(row[6:])  # 第0列为用户ID
            distance = np.linalg.norm(face_feature_vector - db_vector)
            if distance < min_distance:
                min_distance = distance
                matched_user = row[0]
        return matched_user, min_distance

    def show_recognition_result(self, frame):
        if self.matched_user:
            # 获取用户身高信息
            user_row = self.features_df[self.features_df["user_id"] == self.matched_user]
            user_height = user_row["height"].values[0]
            # 在图像上显示用户信息
            cv2.putText(frame, f"User: {self.matched_user}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Height: {user_height} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {self.recognition_confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # 显示座椅调整状态
            if self.seat_controller.is_seat_adjusting:
                cv2.putText(frame, "Adjusting Seat...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "User: Unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    @staticmethod
    def show_instructions(frame):
        cv2.putText(frame, "Press 'Q' or ESC to quit", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'S' to save current user's seat position", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

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
            # 每隔 15帧 进行一次人脸识别
            if frame_count % constants.RECOGNITION_INTERVAL_FRAMES == 0 and not self.seat_controller.is_seat_adjusting:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图进行人脸检测
                faces = self.feature_extractor.detector(gray)
                if len(faces) > 0:  # 如果检测到人脸
                    face = faces[0]  # 只处理检测到的第一张人脸
                    shape = self.feature_extractor.shape_predictor(gray, face)  # 获取人脸关键点
                    # 计算人脸特征向量
                    face_feature_vector = self.feature_extractor.face_recognition_model.compute_face_descriptor(frame, shape)
                    self.matched_user, distance = self.match_face_feature(np.array(face_feature_vector))

                    if distance < constants.RECOGNITION_THRESHOLD:
                        print(f"识别到用户: {self.matched_user}")
                        print(distance)
                        self.seat_controller.current_user_id = self.matched_user
                        self.recognition_confidence = 1.0 - distance
                        # 调整座椅到识别用户的保存位置
                        self.seat_controller.send_user_to_seat_controller()
                    else:
                        print("未识别到匹配的用户")
            # 绘制人脸框
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(self.feature_extractor.detector(gray)) > 0:
                face = self.feature_extractor.detector(gray)[0]
                face_box = (face.left(), face.top(), face.width(), face.height())
                utilities.draw_face_box(frame, face_box)
                if self.matched_user:
                    self.show_recognition_result(frame)
            self.show_instructions(frame)

            # 显示图像
            cv2.imshow("Face Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                print("退出人脸识别")
                break
            if key == ord('s') or key == ord('S') and self.matched_user is not None:
                # 获取当前座椅位置
                seat_position = self.seat_controller.get_current_seat_position()
                if seat_position:
                    # 保存当前用户的座椅位置
                    save_success = self.seat_controller.update_user_seat_position(seat_position)
                    if save_success:
                        print(f"已保存用户 {self.matched_user} 的当前座椅位置")
                    else:
                        print(f"保存用户 {self.matched_user} 的座椅位置失败")
                    cv2.imshow('人脸识别', frame)
                    cv2.waitKey(1000)  # 显示1秒
