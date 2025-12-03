import cv2
import constants
import numpy as np
from typing import List, Tuple


class CameraCalibration:
    def __init__(self):
        self.chessboard_height = constants.CHESSBOARD_HEIGHT
        self.chessboard_width = constants.CHESSBOARD_WIDTH
        self.chessboard_size = (self.chessboard_width, self.chessboard_height)
        self.square_size = constants.SQUARE_SIZE  # 棋盘格方块边长，单位：毫米
        self.image_size = (0, 0)  # 图像大小（分辨率）
        self.image_points = []  # 存储图像点
        self.object_points = []  # 存储物体点
        self.camera_matrix = np.zeros((3, 3), dtype=np.float32)  # 内参矩阵
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 畸变系数
        self.rvecs = []  # 旋转向量
        self.tvecs = []  # 平移向量
        self.rms = 0.0  # 重投影误差
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 终止条件

    def find_corners(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # 寻找棋盘格的角点
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags=flags)
        return ret, corners

    def append_corners(self, image: np.ndarray, corners: np.nanprod) -> np.ndarray:
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 亚像素精度优化
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

        # 绘制识别结果
        cv2.drawChessboardCorners(image, self.chessboard_size, corners, True)

        # 添加结果到 image_points
        self.image_points.append(corners)

        return image

    def calculate(self) -> bool:
        """
        执行相机标定，计算内参矩阵和畸变系数
        Returns:
            bool: 标定是否成功
        """
        # 1. 创建标定板的 3D 参考点（世界坐标系，Z 轴为 0）
        obj = []
        for i in range(self.chessboard_size[1]):
            for j in range(self.chessboard_size[0]):
                obj.append([j * self.square_size, i * self.square_size, 0.0])
        obj = np.array(obj, dtype=np.float32)

        # 2. 生成每幅图像对应的 3D 点
        self.object_points = [obj.copy() for _ in self.image_points]

        # 3. 执行相机标定
        self.camera_matrix = np.zeros((3, 3), dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # 检查 image_size 是否已设置
        if self.image_points and self.image_points[0].shape[0] > 0:
            self.image_size = (self.image_points[0].shape[1], self.image_points[0].shape[0])

        self.rms, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, self.image_size, self.camera_matrix,
            self.dist_coeffs, self.rvecs, self.tvecs, 0, self.criteria
        )

        # 4. 验证标定结果
        result = cv2.checkRange(self.camera_matrix)[0] and cv2.checkRange(self.dist_coeffs)[0]

        # 5. 返回所有结果
        return result


def interactive_calibration():
    capture_time = 0
    capture = cv2.VideoCapture(constants.CAMERA_INDEX)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # 设置摄像头分辨率：W*H = 3840*1080
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_FPS, 30)  # 设置摄像头帧率：30fps
    capture.set(cv2.CAP_PROP_FOURCC, cv2.CAP_OPENCV_MJPEG)  # 设置摄像头编码格式为 MJPEG
    cv2.namedWindow("left", cv2.WINDOW_GUI_EXPANDED)  # 设置窗口大小
    cv2.namedWindow("right", cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("left", 640, 480)
    cv2.resizeWindow("right", 640, 480)
    left_calibrator = CameraCalibration()
    right_calibrator = CameraCalibration()
    while True:
        ret, frame = capture.read()  # 读取一帧图像
        combined_frame = frame.shape[:2]  # 获取图像的高和宽
        if ret:
            # 分割左右图像
            left_frame = frame[:, :combined_frame[1] // 2]
            right_frame = frame[:, combined_frame[1] // 2:]
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("left", left_gray)
            cv2.imshow("right", right_gray)
            # 进行相机标定或测距等
            # ... 此处添加相机标定或测距代码 ...
            res = cv2.waitKey(1)
            if res == 'c' or res == 'C':
                result_left = left_calibrator.find_corners(left_gray)
                result_right = right_calibrator.find_corners(right_gray)
                if result_left[0] and result_right[0]:
                    capture_time += 1
                    print(f"有效帧已捕获！ 当前已捕获 {capture_time} 帧")
                    rms_left = left_calibrator.append_corners(left_gray, result_left[1])
                    rms_right = right_calibrator.append_corners(right_gray, result_right[1])
                    cv2.imshow("left_valid", rms_left)
                    cv2.imshow("right_valid", rms_right)
                else:
                    print("无效帧。")
                    print(f"捕获情况：左侧 -> {result_left[0]}，右侧 -> {result_right[0]}")
                    # 画出左右识别结果
                    cv2.drawChessboardCorners(left_gray, left_calibrator.chessboard_size, result_left[1], result_left[0])
                    cv2.imshow("left_valid", left_gray)
                    cv2.drawChessboardCorners(right_gray, right_calibrator.chessboard_size, result_right[1], result_right[0])
                    cv2.imshow("right_valid", right_gray)
                    cv2.waitKey(1)
            elif res == 'q' or res == 'Q':
                break
            elif res == 'e' or res == 'E':
                left_result = left_calibrator.calculate()
                right_result = right_calibrator.calculate()
                print(left_result)
                print(right_result)
    capture.release()
    cv2.destroyAllWindows()
    return


def main():
    interactive_calibration()

if __name__ == "__main__":
    main()
