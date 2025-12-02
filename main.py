import utilities
import face_registration  # 导入人脸注册模块
import face_feature_extract  # 导入特征提取模块


def main():
    common_context = utilities.CommonContext()  # 创建公共上下文对象
    # -------------- 测试人脸注册功能 -------------- #
    # face_register = face_registration.FaceRegistration(common_context)  # 创建人脸注册对象
    # face_register.register_user_face()  # 调用注册方法
    # -------------- 测试特征提取功能 -------------- #
    feature_extractor = face_feature_extract.FaceFeatureExtractor(common_context)  # 创建特征提取对象
    feature_extractor.extract_features()  # 调用特征提取方法


if __name__ == "__main__":
    main()
