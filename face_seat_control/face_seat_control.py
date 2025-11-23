import time  # 用于时间延迟
import cv2  # 用于图像处理和摄像头操作
import dlib  # 用于人脸检测和特征提取
import pandas as pd  # 用于处理用户数据表格
import serial  # 用于与座椅控制器进行串口通信
from serial.tools import list_ports  # 用于查找可用的串口
import constants.constants as constants  # 导入常量模块


class FaceSeatControl:
    def __init__(self):
        self.current_user_id = None  # 当前用户ID
        self.is_adjusting_seat = False  # 座椅调整状态标志
        self.serial_port = None  # 串口端口
        self.serial_obj = None  # 初始化串口通信
        self.baudrate = constants.BAUDRATE  # 串口波特率
        self.detector = dlib.get_frontal_face_detector()  # 人脸检测器
        self.predictor = dlib.shape_predictor(constants.FACE_LANDMARK_68_PREDICTOR_DAT_PATH)  # 人脸特征点预测器
        self.cap = cv2.VideoCapture(constants.CAMERA_INDEX)  # 打开摄像头

    @staticmethod
    def find_serial_port():
        ports = list_ports.comports()  # 获取所有可用串口列表
        for port in ports:
            print(f"正在检查串口: {port.device}")
            try:
                with serial.Serial(port.device, constants.BAUDRATE, timeout=1, write_timeout=1) as ser:  # 尝试打开串口
                    ser.write((constants.PING_COMMAND + "\r\n").encode())  # 发送PING命令
                    print(f"已发送PING到: {port.device}")
                    response = ser.readline().decode(errors='replace').strip()  # 读取响应
                    print(f"串口响应内容: {response}")
                if constants.PONG_RESPONSE in response:  # 检查响应是否包含PONG
                    # print(f"寻找结果: 成功找到串口: {port.device} !!!")
                    return port.device
                else:
                    print(f"{port.device} 未响应PONG")
            except serial.SerialException as e:
                print(f"{port.device} 连接失败({e}),继续检查下一个端口...")
                continue
        # print(f"寻找结果: 未找到串口!!!")
        return None

    def connect_to_seat_controller(self):
        self.serial_port = self.find_serial_port()  # 查找串口端口
        if self.serial_port is None:
            print(f"寻找结果: 未找到串口!!!")
            return False
        else:
            print(f"寻找结果: 成功找到串口: {self.serial_port} !!!")
            try:
                self.serial_obj = serial.Serial(self.serial_port, self.baudrate, timeout=1, write_timeout=1)  # 打开串口通信
                print(f"已连接到串口: {self.serial_port}")
                return True
            except serial.SerialException as e:
                print(f"连接串口失败: {e}")
                self.serial_obj = None
                return False

    def send_command(self, command):
        if self.serial_obj and self.serial_obj.is_open:
            try:
                self.serial_obj.write((command + "\r\n").encode())  # 发送调整命令
                time.sleep(0.1)  # 等待命令发送完成
                print(f"已发送命令: {command}")
                return True
            except serial.SerialTimeoutException as e:
                print(f"发送命令失败: {e}")
                return False
        else:
            print("串口未连接，无法发送命令。")
            return False

    def get_current_seat_position(self):
        if self.serial_obj is None or not self.serial_obj.is_open:
            print("串口未连接，无法获取座椅位置。")
            return None
        try:
            self.serial_obj.reset_input_buffer()  # 清空接收缓冲区
            self.serial_obj.write((constants.GET_SEAT_POSITION_COMMAND + "\r\n").encode())  # 发送获取座椅位置命令
            time.sleep(0.1)  # 等待响应
            response = self.serial_obj.readline().decode(errors='replace').strip()  # 读取响应
            # --------- 解析响应以获取座椅位置 -----------
            # 响应数据帧格式为:
            position = {}
            if response:
                parts = response.split(',')
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':')
                        position[key.lower()] = int(value)
                # 构造并返回座椅位置字典
                position_result = {
                    "cushion_position": position.get("cushion", 500),
                    "seat_ud_position": position.get("seat_ud", 500),
                    "seat_fb_position": position.get("seat_fb", 500),
                    "backrest_position": position.get("backrest", 500)
                }
                return position_result
        except serial.SerialException as e:
            print(f"获取座椅位置失败: {e}")
            return None

    def update_user_seat_position(self, seat_position):
        try:
            features_df = pd.read_csv(constants.FEATURES_ALL_CSV_PATH)  # 读取用户数据文件
            user_row = features_df[features_df['user_id'] == self.current_user_id]  # 查找当前用户数据
            if user_row.empty:
                print(f"未找到用户ID {self.current_user_id} 的数据。")
                return False
            # 更新座椅位置数据
            current_user_index = user_row.index[0]
            features_df.at[current_user_index, 'cushion_position'] = seat_position.get('cushion_position', 500)
            features_df.at[current_user_index, 'seat_ud_position'] = seat_position.get('seat_ud_position', 500)
            features_df.at[current_user_index, 'seat_fb_position'] = seat_position.get('seat_fb_position', 500)
            features_df.at[current_user_index, 'backrest_position'] = seat_position.get('backrest_position', 500)
            features_df.to_csv(constants.FEATURES_ALL_CSV_PATH, index=False)  # 保存更新后的数据
            print(f"已更新用户ID {self.current_user_id} 的座椅位置数据。")
            # 同时更新用户信息CSV文件
            user_info_df = pd.read_csv(constants.USER_INFO_CSV_PATH)
            user_info_row = user_info_df[user_info_df['user_id'] == self.current_user_id]
            if not user_info_row.empty:
                current_user_info_index = user_info_row.index[0]
                user_info_df.at[current_user_info_index, 'cushion_position'] = seat_position.get('cushion_position', 500)
                user_info_df.at[current_user_info_index, 'seat_ud_position'] = seat_position.get('seat_ud_position', 500)
                user_info_df.at[current_user_info_index, 'seat_fb_position'] = seat_position.get('seat_fb_position', 500)
                user_info_df.at[current_user_info_index, 'backrest_position'] = seat_position.get('backrest_position', 500)
                user_info_df.to_csv(constants.USER_INFO_CSV_PATH, index=False)
                print(f"已更新用户ID {self.current_user_id} 的用户信息数据。")
            # 保存到座椅控制器
            self.send_command(constants.SAVE_POSITION_COMMAND)
            print(f"已保存用户ID {self.current_user_id} 的座椅位置到控制器。")
            return True
        except Exception as e:
            print(f"更新用户座椅位置失败: {e}")
            return False

    def send_user_to_seat_controller(self):
        # 构建座椅调整命令
        command = f"USER_ID:{self.current_user_id},"
        command+= f"CUSHION:{}"

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()  # 释放摄像头资源
        if self.serial_obj and self.serial_obj.is_open:
            self.serial_obj.close()  # 关闭串口通信


def main():
    face_seat_control = FaceSeatControl()  # 创建FaceSeatControl实例
    try:
        face_seat_control.connect_to_seat_controller()  # 连接到座椅控制器
        face_seat_control.send_command("ADJUST_SEAT_TO_USER_1")  # 发送调整命令示例
        face_seat_control.get_current_seat_position()
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()  # 运行主函数
