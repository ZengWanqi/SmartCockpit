import os
import cv2
import dlib
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import constants


class FaceFeatureExtractor:
    def __init__(self):
        print("加载人脸检测和特征提取模型...")
        self.detector = dlib.get_frontal_face_detector() # 使用dlib自带的人脸检测器
        self.shape_predictor = dlib.shape_predictor(constants.FACE_LANDMARK_68_PREDICTOR_DAT_PATH) # 加载68点人脸标志预测模型
        self.face_recognition_model = dlib.face_recognition_model_v1(constants.FACE_RECOGNITION_RESNET_MODEL_V1_DAT_PATH) # 加载人脸识别模型
        self.data_dir = constants.DATA_DIR_PATH # 数据目录路径
        self.user_info = self.load_user_info() # 加载用户信息
        print(f"成功加载 {len(self.user_info)} 条用户信息")

    @staticmethod
    def load_user_info():
        user_info = {}
        # 检查用户信息CSV文件是否存在
        if not os.path.exists(constants.USER_INFO_CSV_PATH):
            print(f"警告：用户信息文件 {constants.USER_INFO_CSV_PATH} 不存在")
            return user_info
        with open(constants.USER_INFO_CSV_PATH, "r") as csvfile:
            reader = csv.reader(csvfile)
            try:
                next(reader)  # 跳过表头
            except StopIteration:
                print("警告：用户信息CSV文件为空")
                return user_info
            for row in reader:
                if len(row) >= 6:  # 确保有足够的字段
                    user_info[row[0]] = {
                        "height": row[1],
                        "cushion_position": row[2],
                        "seat_ud_position": row[3],
                        "seat_fb_position": row[4],
                        "backrest_position": row[5]
                    }
        return user_info

    def extract_features(self):
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            print(f"错误：数据目录 {self.data_dir} 不存在")
            return []

        user_folders = [f for f in os.listdir(self.data_dir)
                        if os.path.isdir(os.path.join(self.data_dir, f)) and f != ".git"]
        print(f"找到 {len(user_folders)} 个用户文件夹")

        all_features = []
        for user_id in tqdm(user_folders, desc="处理用户"):
            user_path = os.path.join(self.data_dir, user_id)
            face_images = [f for f in os.listdir(user_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"用户 {user_id}：找到 {len(face_images)} 张图像")

            if not face_images:
                print(f"警告: 用户 {user_id} 没有找到人脸图像，跳过")
                continue

            user_features = []
            for img_name in face_images:
                img_path = os.path.join(user_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告: 无法读取图像 {img_path}，跳过")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = self.detector(gray, 1)
                if len(dets) == 0:
                    print(f"警告: 在图像 {img_path} 中未检测到人脸，跳过")
                    continue

                # 只取第一张检测到的人脸（避免多脸干扰）
                d = dets[0]
                shape = self.shape_predictor(gray, d)
                face_descriptor = self.face_recognition_model.compute_face_descriptor(img, shape)
                user_features.append(np.array(face_descriptor))

            if user_features:
                mean_feature = np.mean(user_features, axis=0)
                feature_record = {
                    "user_id": user_id,
                    "feature": mean_feature
                }
                # 补充用户信息（无信息则留空）
                feature_record.update(self.user_info.get(user_id, {
                    "height": "",
                    "cushion_position": "",
                    "seat_ud_position": "",
                    "seat_fb_position": "",
                    "backrest_position": ""
                }))
                all_features.append(feature_record)
                print(f"用户 {user_id}：成功提取 {len(user_features)} 张图像的特征，已保存平均特征")
            else:
                print(f"警告: 用户 {user_id} 没有有效特征，跳过")

        print(f"\n特征提取完成：共获取 {len(all_features)} 个用户的有效特征")
        return all_features

    @staticmethod
    def save_features_to_csv(features, output_csv_path):
        # 新增：判断特征列表是否为空
        if not features:
            print("警告：没有提取到任何人脸特征，无需保存CSV")
            return

        df = pd.DataFrame(features)
        # 确保特征数组长度一致
        feature_len = len(features[0]['feature'])
        feature_columns = [f"feature_{i}" for i in range(feature_len)]
        feature_data = pd.DataFrame(df['feature'].tolist(), columns=feature_columns)

        df = pd.concat([df.drop(columns=['feature']), feature_data], axis=1)
        df.to_csv(output_csv_path, index=False)
        print(f"特征已成功保存到 {output_csv_path}")


if __name__ == "__main__":
    extractor = FaceFeatureExtractor()
    features = extractor.extract_features()
    extractor.save_features_to_csv(features, constants.FEATURES_ALL_CSV_PATH)