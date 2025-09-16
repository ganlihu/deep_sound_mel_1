import logging
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder  # 新增：导入LabelEncoder
from chewbite_fusion.models.deep_sound import DeepSound
from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features.feature_factories import FeatureFactory_RawAudioData
from chewbite_fusion.data.utils import NaNDetector

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    return DeepSound(
        input_size=1800,
        output_size=4,  # 确保与标签类别数一致（含no-event）
        n_epochs=2,
        batch_size=8,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True,
        optimizer_kwargs={
            'learning_rate': 5e-5,
            'clipnorm': 2.0,
            'clipvalue': 1.0
        }
    )


@experiment()
def deep_sound():
    """ Experiment with Deep Sound architecture. """
    # 注释或删除这行：tf.debugging.enable_check_numerics() （与XLA GPU不兼容）
    
    # 设置全局数值精度（避免浮点溢出）
    tf.keras.backend.set_floatx('float32')
    
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(
        window_width=window_width,
        window_overlap=window_overlap,
        include_movement_magnitudes=False,
        audio_sampling_frequency=6000
    )
    
    nan_detector = NaNDetector(verbose=True)
    X_zavalla = X['zavalla2022']
    y_zavalla = y['zavalla2022']  # y_zavalla的标签是字符串类型（如'chew'、'no-event'）
    
    # 检查X数据
    try:
        for key, segments in X_zavalla.items():
            nan_detector.check_nan(segments, f"X片段 {key}")
        logger.info("X数据中未检测到NaN值")
    except ValueError as e:
        logger.error(f"X数据异常: {str(e)}")
        raise
    
    # 收集所有标签并编码为整数
    try:
        all_labels = []
        for key, labels in y_zavalla.items():
            nan_detector.check_nan(labels, f"y标签 {key}")  # 检查标签中是否有NaN
            all_labels.extend(labels)  # 收集所有字符串标签
        
        # 步骤1：初始化LabelEncoder并拟合所有标签
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)  # 学习字符串到整数的映射（如'no-event'→0，'chew'→1等）
        logger.info(f"标签映射关系: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
        # 步骤2：将所有y标签从字符串转换为整数
        y_encoded = {}  # 存储编码后的整数标签
        for key, labels in y_zavalla.items():
            y_encoded[key] = label_encoder.transform(labels)  # 转换当前片段的标签
        y['zavalla2022'] = y_encoded  # 替换原y中的字符串标签为编码后的整数标签
        
        # 步骤3：验证编码后的标签（确保是整数且范围正确）
        all_encoded_labels = []
        for labels in y_encoded.values():
            all_encoded_labels.extend(labels)
        unique_encoded_labels = np.unique(all_encoded_labels)
        
        if not np.issubdtype(unique_encoded_labels.dtype, np.integer):
            raise ValueError(f"编码后标签仍非整数，实际类型: {unique_encoded_labels.dtype}")
        
        n_classes = len(label_encoder.classes_)
        if np.any(unique_encoded_labels < 0) or np.any(unique_encoded_labels >= n_classes):
            raise ValueError(f"编码后标签超出范围[0, {n_classes-1}]，实际值: {unique_encoded_labels}")
        
        logger.info(f"标签编码成功，有效整数标签: {unique_encoded_labels}，类别数: {n_classes}")
    except ValueError as e:
        logger.error(f"y标签异常: {str(e)}")
        raise
    
    # 特征缩放（保持不变）
    logger.info("应用特征缩放...")
    X_scaled = {}
    for key, segments in X_zavalla.items():
        seg_array = np.array(segments, dtype=np.float32)
        seg_max = np.max(np.abs(seg_array)) + 1e-8
        seg_scaled = seg_array / seg_max
        X_scaled[key] = seg_scaled
        logger.info(f"片段 {key} 缩放后范围: [{np.min(seg_scaled):.4f}, {np.max(seg_scaled):.4f}]")
    X['zavalla2022'] = X_scaled
    
    # 日志：数据基本信息
    logger.info("生成的片段数量: %s", len(X_zavalla.keys()))
    logger.info("片段编号示例: %s", list(X_zavalla.keys())[:5])
    
    # 初始化实验（禁用数据增强）
    e = Experiment(
        get_model_instance,
        FeatureFactory_RawAudioData,
        X, y,  # 此时y中的标签已为整数
        window_width=window_width,
        window_overlap=window_overlap,
        name='deep_sound',
        manage_sequences=True,
        use_raw_data=True,
        data_augmentation=False
    )
    
    # 训练过程中监控NaN（自定义回调）
    class NaNMonitor(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs is not None:
                for k, v in logs.items():
                    if np.isnan(v) or np.isinf(v):
                        logger.error(f"批次 {batch} 出现异常值: {k}={v}")
                        self.model.stop_training = True
    
    # 修正：通过模型工厂传递回调（避免使用e.add_callbacks）
    def model_factory_with_callbacks(variable_params):
        model = get_model_instance(variable_params)
        model.callbacks = [NaNMonitor()]  # 假设模型训练时会使用callbacks属性
        return model
    
    # 重新初始化实验，使用带回调的模型工厂
    e = Experiment(
        model_factory_with_callbacks,
        FeatureFactory_RawAudioData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deep_sound',
        manage_sequences=True,
        use_raw_data=True,
        data_augmentation=False
    )
    
    e.run()