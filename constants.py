# ------------constants.py----------
# 人脸头像框的颜色 (BGR 格式)
FACE_BOX_COLOR = (0, 255, 0)  # 绿色

# 数据目录名称
DATA_DIR_NAME = "data"

# 数据目录路径
DATA_DIR_PATH = "data"

# 用户信息 CSV 文件名称
USER_INFRO_CSV_FILE_NAME = "user_info.csv"

# 用户信息 CSV 文件路径
USER_INFO_CSV_PATH = "data/user_info.csv"

# CSV 文件头
CSV_HEADERS = ["user_id", "height", "cushion_position", "seat_ud_position", "seat_fb_position", "backrest_position"]

# 采集图像的默认数量
DEFAULT_IMAGE_COUNT = 10

# 摄像头索引，默认使用第一个摄像头
CAMERA_INDEX = 0

# ------------face_feature_extract_constants.py----------

# 68点人脸关键点预测模型文件
FACE_LANDMARK_68_PREDICTOR_DAT = 'shape_predictor_68_face_landmarks.dat'

# 68点人脸关键点预测模型文件路径
FACE_LANDMARK_68_PREDICTOR_DAT_PATH = 'resources/shape_predictor_68_face_landmarks.dat'

# Dlib人脸识别ResNet模型文件
FACE_RECOGNITION_RESNET_MODEL_V1_DAT = 'dlib_face_recognition_resnet_model_v1.dat'

# Dlib人脸识别ResNet模型文件路径
FACE_RECOGNITION_RESNET_MODEL_V1_DAT_PATH = 'resources/dlib_face_recognition_resnet_model_v1.dat'

# 所有用户特征CSV文件路径
FEATURES_ALL_CSV_PATH = 'data/features_all.csv'

# --------------face_seat_control_constants.py----------

# 串口波特率
BAUDRATE = 115200

# PING命令，发送用于检测设备是否在线
PING_COMMAND = "PING"

# PONG响应，设备在线时返回的响应
PONG_RESPONSE = "PONG"

# 获取当前座椅位置命令
GET_SEAT_POSITION_COMMAND = "GET_SEAT_POSITION"

# 保存当前座椅位置命令
SAVE_POSITION_COMMAND =  "SAVE_POSITION"

# 保存位置确认响应
SAVE_POSITION_CONFIRM_RESPONSE = "POSITION_SAVED"

# 座椅调整确认响应
SEAT_ADJUSTMENT_CONFIRM_RESPONSE = "SEAT_ADJUSTED"

# 座椅调整超时时间，单位为秒
SEAT_ADJUSTMENT_TIMEOUT = 30

# 每隔多少帧检测一次人脸
RECOGNITION_INTERVAL_FRAMES = 15

# 人脸识别距离阈值
RECOGNITION_THRESHOLD = 0.6