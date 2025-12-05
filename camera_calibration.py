import cv2
import constants
import numpy as np
from typing import Tuple, Optional


class CameraCalibration:
    def __init__(self):
        self.chessboard_height = constants.CHESSBOARD_HEIGHT
        self.chessboard_width = constants.CHESSBOARD_WIDTH
        self.chessboard_size = (self.chessboard_width, self.chessboard_height)  # 棋盘格内角点数量 (宽, 高)
        self.square_size = constants.SQUARE_SIZE  # 棋盘格方块边长，单位：毫米
        self.image_size = (0, 0)  # 图像大小（分辨率）（宽, 高）
        self.image_points = []  # 存储图像点
        self.object_points = []  # 存储物体点
        self.camera_matrix = np.zeros((3, 3), dtype=np.float32)  # 内参矩阵
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 畸变系数
        self.rvecs = []  # 旋转向量
        self.tvecs = []  # 平移向量
        self.rms_repro_error = 0.0  # 重投影误差
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 终止条件

    def find_corners(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        寻找棋盘格的角点
        :param gray:  输入灰度图像
        :return:
            bool: 是否找到角点
            np.ndarray or None: 如果找到角点（bool为True），则为包含角点坐标的 np.ndarray；
                                如果未找到角点（bool为False），则可能为 None 或空的 np.ndarray。
        """

        # 检查输入图像是否有效
        if gray is None or gray.size == 0:
            return False, None

        try:
            # 寻找棋盘格的角点
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
            is_corner_found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags=flags)
            return is_corner_found, corners
        except cv2.error as e:
            print(f"OpenCV 错误: {e}")
            return False, None

    def append_corners(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        添加角点到图像点列表，并绘制识别结果
        :param gray: 输入灰度图像
        :param corners: 角点坐标
        :return:
            np.ndarray: 绘制了识别结果的RGB图像
        """

        # 输入验证
        if gray is None or gray.size == 0:
            raise ValueError("输入的灰度图像无效")
        if corners is None or corners.size == 0:
            raise ValueError("输入的角点坐标无效")

        if self.image_size == (0, 0):
            self.image_size = gray.shape[::-1]  # 图像大小（分辨率）: (宽, 高)
        # 亚像素精度优化
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

        # 绘制识别结果
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(frame, self.chessboard_size, corners, True)

        # 添加结果到 image_points
        self.image_points.append(corners)

        return frame  # 返回绘制了识别结果的图像(彩色图)

    def calculate(self) -> bool:
        """
        执行相机标定，计算内参矩阵和畸变系数
        Returns:
            bool: 标定是否成功
        """
        # 1. 生成单幅图像的 3D 点
        obj = []  # 存储单幅图像的 3D 点
        for i in range(self.chessboard_size[1]):  # 高度方向
            for j in range(self.chessboard_size[0]):  # 宽度方向
                obj.append([j * self.square_size, i * self.square_size, 0.0])
        obj = np.array(obj, dtype=np.float32)

        # 2. 生成每幅图像对应的 3D 点
        self.object_points = [obj.copy() for _ in self.image_points]

        # 3. 执行相机标定
        self.rms_repro_error, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, self.image_size, self.camera_matrix,
            self.dist_coeffs, self.rvecs, self.tvecs, 0, self.criteria
        )

        # 4. 验证标定结果
        result = cv2.checkRange(self.camera_matrix)[0] and cv2.checkRange(self.dist_coeffs)[0]

        # 5. 返回所有结果
        return result


def interactive_calibration():
    capture_time = 0

    # ------------- 初始化摄像头 -------------
    capture = cv2.VideoCapture(constants.CAMERA_INDEX)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # 设置摄像头分辨率：W*H = 3840*1080
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_FPS, 30)  # 设置摄像头帧率：30fps
    capture.set(cv2.CAP_PROP_FOURCC, cv2.CAP_OPENCV_MJPEG)  # 设置摄像头编码格式为 MJPEG
    cv2.namedWindow("left", cv2.WINDOW_GUI_EXPANDED)  # 设置窗口大小
    cv2.namedWindow("right", cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("left", 640, 480)
    cv2.resizeWindow("right", 640, 480)

    # ------------- 初始化相机标定器 -------------
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
            cv2.resizeWindow("left", 640, 480)
            cv2.imshow("right", right_gray)
            cv2.resizeWindow("right", 640, 480)

            res = cv2.waitKey(1)

            if res == ord('c') or res == ord('C'):  # 拍摄
                result_left = left_calibrator.find_corners(left_gray)
                result_right = right_calibrator.find_corners(right_gray)

                if result_left[0] and result_right[0]:  # 如果左右都找到角点
                    capture_time += 1
                    print(f"有效帧已捕获！ 当前已捕获 {capture_time} 帧")
                    if capture_time >= 5:
                        print("已达到所需捕获帧数，按 'E' 键计算标定参数，或继续捕获更多帧以提高精度。")
                    # 添加角点并显示结果
                    left_drawn = left_calibrator.append_corners(left_gray, result_left[1])
                    right_drawn = right_calibrator.append_corners(right_gray, result_right[1])

                    cv2.imshow("left_valid", cv2.resize(left_drawn, (640, 480)))
                    cv2.imshow("right_valid", cv2.resize(right_drawn, (640, 480)))
                else:
                    print("无效帧。")
                    print(f"捕获情况：左侧 -> {result_left[0]}，右侧 -> {result_right[0]}")
                    # 画出左右识别结果
                    cv2.drawChessboardCorners(left_frame, left_calibrator.chessboard_size, result_left[1], result_left[0])
                    cv2.imshow("left_invalid", cv2.resize(left_frame, (640, 480)))
                    cv2.drawChessboardCorners(right_frame, right_calibrator.chessboard_size, result_right[1], result_right[0])
                    cv2.imshow("right_invalid", cv2.resize(right_frame, (640, 480)))
                    cv2.waitKey(1)
            elif res == ord('q') or res == ord('Q'):
                break
            elif res == ord('e') or res == ord('E'):
                left_result = left_calibrator.calculate()
                right_result = right_calibrator.calculate()
                print(left_result)
                print(right_result)
    capture.release()
    cv2.destroyAllWindows()
    print(f"----------------------------单目标定结果----------------------------")
    print(f"左相机：")
    print(f"重投影误差 RMS = {left_calibrator.rms_repro_error}")
    print(f"内参矩阵 = \n{left_calibrator.camera_matrix}")
    print(f"畸变系数 = \n{left_calibrator.dist_coeffs.ravel()}")
    print(f"右相机：")
    print(f"重投影误差 RMS = {right_calibrator.rms_repro_error}")
    print(f"内参矩阵 = \n{right_calibrator.camera_matrix}")
    print(f"畸变系数 = \n{right_calibrator.dist_coeffs.ravel()}")

    # ------------------------下面进行双目标定------------------------
    obj = []  # 存储单幅图像的 3D 点
    for i in range(left_calibrator.chessboard_size[1]):  # 高度方向
        for j in range(left_calibrator.chessboard_size[0]):  # 宽度方向
            obj.append([j * left_calibrator.square_size, i * left_calibrator.square_size, 0.0])
    obj = np.array(obj, dtype=np.float32)
    object_points = [obj.copy() for _ in range(capture_time)]

    # ------------------------ 执行双目标定 ------------------------

    # 初始化初始变量
    R = np.zeros((3, 3), dtype=np.float32)  # 旋转矩阵
    T = np.zeros((3, 1), dtype=np.float32)  # 平移向量
    essential_matrix = np.zeros((3, 3), dtype=np.float32)  # 本质矩阵
    fundamental_matrix = np.zeros((3, 3), dtype=np.float32)  # 基础矩阵

    # 执行双目标定
    rms, camera_matrix1, distcoeffs1, camera_matrix2, distcoeffs2, R, T, essential_matrix, fundamental_matrix = cv2.stereoCalibrate(
        object_points,
        left_calibrator.image_points, right_calibrator.image_points,
        left_calibrator.camera_matrix, left_calibrator.dist_coeffs,
        right_calibrator.camera_matrix, right_calibrator.dist_coeffs,
        left_calibrator.image_size,
        R, T,
        essential_matrix,
        fundamental_matrix,
        cv2.CALIB_FIX_INTRINSIC,
        left_calibrator.criteria
    )

    # 输出双目标定结果
    print(f"----------------------------双目标定结果----------------------------")
    print(f"重投影误差 RMS = {rms}")
    print(f"旋转矩阵 R = \n{R}")
    print(f"平移向量 T = \n{T.ravel()}")
    print(f"本质矩阵 = \n{essential_matrix}")
    print(f"基础矩阵 = \n{fundamental_matrix}")

    # 保存标定结果到文件
    np.savez(constants.CAMERA_CALIBRATION_RESULT_PATH,
             left_camera_matrix=left_calibrator.camera_matrix,
             left_dist_coeffs=left_calibrator.dist_coeffs,
             right_camera_matrix=right_calibrator.camera_matrix,
             right_dist_coeffs=right_calibrator.dist_coeffs,
             R=R,
             T=T,
             essential_matrix=essential_matrix,
             fundamental_matrix=fundamental_matrix)
    print(f"标定结果已保存到文件：{constants.CAMERA_CALIBRATION_RESULT_PATH}")
    return


def show_calibration_parameters():
    data = np.load(constants.CAMERA_CALIBRATION_RESULT_PATH)
    print("左相机内参矩阵：")
    print(data['left_camera_matrix'])
    print("左相机畸变系数：")
    print(data['left_dist_coeffs'].ravel())
    print("右相机内参矩阵：")
    print(data['right_camera_matrix'])
    print("右相机畸变系数：")
    print(data['right_dist_coeffs'].ravel())
    print("旋转矩阵 R：")
    print(data['R'])
    print("平移向量 T：")
    print(data['T'].ravel())
    print("本质矩阵：")
    print(data['essential_matrix'])
    print("基础矩阵：")
    print(data['fundamental_matrix'])


def main():
    interactive_calibration()
    show_calibration_parameters()


if __name__ == "__main__":
    main()
