from pathlib import Path  # 用于路径操作
import logging  # 用于日志记录
import csv  # 用于 CSV 文件读写（存储用户信息）
import cv2  # 摄像头操作和图像处理
import dlib  # 人脸检测
import shutil  # 用于文件夹删除操作
from datetime import datetime  # 用于生成时间戳
import constants  # 导入常量


class FaceRegistration:
    def __init__(self):
        self.init_workspace()

    @staticmethod
    def init_workspace() -> None:
        """
        初始化数据目录并确保  CSV 文件存在且包含表头。
        """
        data_dir = Path(constants.DATA_DIR_PATH)
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            csv_path = data_dir / constants.USER_INFRO_CSV_FILE_NAME
            if not csv_path.exists():
                with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(constants.CSV_HEADERS)
        except Exception as e:
            logging.error(f"初始化工作区失败: {e}")

    @staticmethod
    def collect_face_data(max_count: int = 20) -> dict:
        """
        采集用户人脸数据并记录到 CSV。
        参数:
            max_count: 需要采集的图像数量。
        返回:
            dict: 采集结果信息。
        """
        data_dir = Path(constants.DATA_DIR_PATH)
        user_id = input("请输入用户ID (例如: 12345): ").strip()
        if not user_id:
            print("用户ID不能为空")
            return {"success": False, "reason": "empty_user_id"}
        csv_path = data_dir / constants.USER_INFRO_CSV_FILE_NAME
        user_folder = data_dir / user_id
        user_height = input("请输入身高 (cm): ").strip()
        if not user_height.isdigit():
            print("身高需为整型数字")
            return {"success": False, "reason": "invalid_height"}
        if user_folder.exists():
            confirm = input(f"用户 {user_id} 已存在，是否覆盖? (y/n): ").lower().strip()
            if confirm == 'y':
                shutil.rmtree(user_folder)
                if csv_path.exists():  # 确保CSV文件存在
                    # 读取所有记录
                    with csv_path.open("r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        headers = next(reader)  # 保存表头
                        rows = [row for row in reader if row[0] != user_id]  # 过滤旧记录（row[0]是user_id）
                    # 写回过滤后的记录（即删除旧记录）
                    with csv_path.open("w", encoding="utf-8", newline="") as f:
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
            csv_path = data_dir / constants.USER_INFRO_CSV_FILE_NAME
            with csv_path.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    user_id,
                    user_height,
                    default_seat_positions["cushion_position"],
                    default_seat_positions["seat_ud_position"],
                    default_seat_positions["seat_fb_position"],
                    default_seat_positions["backrest_position"]
                ])
        except Exception as e:
            logging.error(f"写入 CSV 失败: {e}")
            return {"success": False, "reason": "csv_write_failed"}

        print("开始采集人脸数据，请保持面部在摄像头中")
        print("按 ESC 取消采集")

        cap = cv2.VideoCapture(constants.CAMERA_INDEX)
        if not cap.isOpened():
            print("摄像头未能打开")
            return {"success": False, "reason": "camera_open_failed"}

        detector = dlib.get_frontal_face_detector()

        count = 0
        frame_counter = 0

        try:
            while count < max_count:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取摄像头画面")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                cv2.putText(frame, f"Collected: {count}/{max_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if faces:
                    frame_counter += 1
                    face = faces[0]
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    padding = 20  # 可调整的扩展像素
                    cv2.rectangle(frame, (max(x - padding, 0), max(y - padding, 0)),
                                  (min(x + w + padding, frame.shape[1]), min(y + h + padding, frame.shape[0])),
                                  (0, 255, 0), 2)

                    if count < max_count and frame_counter % 5 == 0:
                        y0, y1 = max(y, 0), max(y + h, 0)
                        x0, x1 = max(x, 0), max(x + w, 0)
                        face_img = frame[y0:y1, x0:x1]
                        if face_img.size > 0:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            img_path = user_folder / f"{timestamp}.jpg"
                            cv2.imwrite(str(img_path), face_img)
                            count += 1
                            print(f"已保存 {count}/{max_count} 张照片")

                cv2.imshow('人脸采集', frame)
                key = cv2.waitKey(1)
                if key == 27:
                    print("已取消采集")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if count >= max_count:
            print(f"已完成用户 {user_id} 的人脸采集")
            return {"success": True, "user_id": user_id, "images": count}
        else:
            print(f"用户 {user_id} 的人脸采集未完成")
            return {"success": False, "user_id": user_id, "images": count}


def main():
    registrar = FaceRegistration()
    result = registrar.collect_face_data(constants.DEFAULT_IMAGE_COUNT)
    print(result)


if __name__ == "__main__":
    main()
