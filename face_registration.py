from pathlib import Path  # 用于路径操作
import utilities  # 导入自定义工具模块
import logging  # 用于日志记录
import csv  # 用于 CSV 文件读写（存储用户信息）
import cv2  # 摄像头操作和图像处理
import dlib  # 人脸检测
import shutil  # 用于文件夹删除操作
import constants  # 导入常量


class FaceRegistration:
    def __init__(self, common_context: utilities.CommonContext):
        self.data_path = Path(constants.DATA_DIR_PATH)  # 数据目录路径
        self.user_info_csv_path = self.data_path / constants.USER_INFRO_CSV_FILE_NAME  # 用户信息 CSV 文件路径
        self.camera = common_context.camera  # 使用公共上下文中的摄像头对象
        self.detector = common_context.detector  # 使用公共上下文中的人脸检测器
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            if not self.user_info_csv_path.exists():
                with self.user_info_csv_path.open("w", encoding="utf-8", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(constants.CSV_HEADERS)
        except Exception as e:
            logging.error(f"初始化工作区失败: {e}")

    def register_user_face(self, max_count: int = 20) -> dict:
        """
        注册用户人脸数据
        :param max_count: 采集的最大人脸图像数量
        :return: 注册结果字典
        例如: {"success": True, "user_id": "user123", "images
        """
        register_user = input("请输入用户ID: ").strip()
        if not register_user:
            print("用户ID不能为空")
            return {"success": False, "reason": "empty_user_id"}
        user_folder = self.data_path / register_user
        user_height = input("请输入身高 (cm): ").strip()
        if not user_height.isdigit():
            print("身高需为整型数字")
            return {"success": False, "reason": "invalid_height"}
        if user_folder.exists():
            confirm = input(f"用户 {register_user} 已存在，是否覆盖? (y/n): ").lower().strip()
            if confirm == 'y':
                shutil.rmtree(user_folder)
                if self.user_info_csv_path.exists():  # 确保CSV文件存在
                    # 读取所有记录
                    with self.user_info_csv_path.open("r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        headers = next(reader)  # 保存表头
                        rows = [row for row in reader if row[0] != register_user]  # 过滤旧记录（row[0]是user_id）
                    # 写回过滤后的记录（即删除旧记录）
                    with self.user_info_csv_path.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)  # 写表头
                        writer.writerows(rows)  # 写其他用户记录
            else:
                print("已取消操作")
                return {"success": False, "reason": "user_exists_cancel"}
        user_folder.mkdir(parents=True, exist_ok=True)
        default_seat_positions = {
            "cushion_position": 500,  # 默认座垫位置
            "seat_ud_position": 500,  # 默认座椅上下位置
            "seat_fb_position": 500,  # 默认座椅前后位置
            "backrest_position": 500  # 默认靠背位置
        }
        # 写入 CSV
        try:
            with self.user_info_csv_path.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([register_user, user_height, default_seat_positions["cushion_position"], default_seat_positions["seat_ud_position"],
                                 default_seat_positions["seat_fb_position"], default_seat_positions["backrest_position"]])
        except Exception as e:
            logging.error(f"写入 CSV 失败: {e}")
            return {"success": False, "reason": "csv_write_failed"}
        print("开始采集人脸数据，请保持面部在摄像头中")
        print("按 ESC 取消采集")
        if not self.camera.isOpened():
            print("摄像头未能打开")
            return {"success": False, "reason": "camera_open_failed"}
        count = 0
        frame_counter = 0
        try:
            while count < max_count:
                ret, frame = self.camera.read()
                if not ret:
                    print("无法获取摄像头画面")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                cv2.putText(frame, f"Collected: {count}/{max_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if faces:
                    frame_counter += 1
                    face = faces[0]
                    face_box = (face.left(), face.top(), face.width(), face.height())
                    utilities.draw_face_box(frame, face_box)  # 绘制人脸框
                    if count < max_count and frame_counter % 5 == 0:  # 每隔5帧保存一次人脸图像
                        utilities.save_face_image(frame, face_box, user_folder, register_user)
                        count += 1
                        print(f"已保存 {count}/{max_count} 张照片")
                cv2.imshow('人脸采集', frame)
                key = cv2.waitKey(1)
                if key == 27: # ESC 键
                    print("已取消采集")
                    break
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

        if count >= max_count:
            print(f"已完成用户 {register_user} 的人脸采集")
            return {"success": True, "user_id": register_user, "images": count}
        else:
            print(f"用户 {register_user} 的人脸采集未完成")
            return {"success": False, "user_id": register_user, "images": count}
