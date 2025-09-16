import logging
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from chewbite_fusion.models.deep_sound import DeepSound
from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features.feature_factories import FeatureFactory_RawAudioData
from chewbite_fusion.data.utils import NaNDetector

from yaer.base import experiment


logger = logging.getLogger('yaer')


# 核心修改：接收num_classes参数，动态匹配输出维度
def get_model_instance(variable_params, num_classes=5):
    return DeepSound(
        input_size=1800,
        output_size=num_classes,  # 与实际类别数一致
        n_epochs=2,
        batch_size=8,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True,
        optimizer_kwargs={
            'learning_rate': 1e-5,  # 降低学习率，减少震荡
            'clipnorm': 5.0,        # 增强梯度裁剪
            'clipvalue': 2.0
        }
    )


@experiment()
def deep_sound():
    """ Experiment with Deep Sound architecture. """
    # 启用数值检查，快速定位NaN来源
    tf.debugging.enable_check_numerics()
    
    # 设置全局数值精度
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
    y_zavalla = y['zavalla2022']  # 字符串标签
    
    # 检查X数据
    try:
        for key, segments in X_zavalla.items():
            nan_detector.check_nan(segments, f"X片段 {key}")
        logger.info("X数据中未检测到NaN值")
    except ValueError as e:
        logger.error(f"X数据异常: {str(e)}")
        raise
    
    # 标签编码（字符串→整数）
    try:
        all_labels = []
        for key, labels in y_zavalla.items():
            nan_detector.check_nan(labels, f"y标签 {key}")
            all_labels.extend(labels)
        
        # 标签编码
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        logger.info(f"标签映射关系: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
        # 转换所有标签
        y_encoded = {}
        for key, labels in y_zavalla.items():
            y_encoded[key] = label_encoder.transform(labels)
        y['zavalla2022'] = y_encoded
        
        # 验证编码结果
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
    
    # 核心修改：特征缩放改为标准化（更稳定）
    logger.info("应用特征标准化...")
    X_scaled = {}
    for key, segments in X_zavalla.items():
        seg_array = np.array(segments, dtype=np.float32)
        
        # 异常值处理：截断3倍标准差外的值
        mean = np.mean(seg_array)
        std = np.std(seg_array)
        seg_array = np.clip(seg_array, mean - 3*std, mean + 3*std)
        
        # 标准化（均值0，标准差1）
        scaler = StandardScaler()
        # 处理三维数据（[num_windows, length, channels] → 展平为二维）
        num_windows, length, channels = seg_array.shape
        seg_reshaped = seg_array.reshape(-1, channels)  # [num_windows*length, channels]
        seg_scaled_reshaped = scaler.fit_transform(seg_reshaped)
        seg_scaled = seg_scaled_reshaped.reshape(num_windows, length, channels)  # 恢复形状
        
        X_scaled[key] = seg_scaled
        logger.info(f"片段 {key} 标准化后范围: [{np.min(seg_scaled):.4f}, {np.max(seg_scaled):.4f}]")
    X['zavalla2022'] = X_scaled
    
    # 日志：数据基本信息
    logger.info("生成的片段数量: %s", len(X_zavalla.keys()))
    logger.info("片段编号示例: %s", list(X_zavalla.keys())[:5])
    
    # 初始化实验
    e = Experiment(
        get_model_instance,
        FeatureFactory_RawAudioData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deep_sound',
        manage_sequences=True,
        use_raw_data=True,
        data_augmentation=False  # 先禁用增强，排查问题
    )
    
    # 训练过程中监控NaN（增强版）
    class NaNMonitor(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs is not None:
                for k, v in logs.items():
                    if np.isnan(v) or np.isinf(v):
                        logger.error(f"批次 {batch} 出现异常值: {k}={v}")
                        # 记录最后批次数据用于调试
                        try:
                            batch_x = self.model.train_function.inputs[0].numpy()
                            batch_y = self.model.train_function.inputs[1].numpy()
                            np.save(f"nan_batch_x_{batch}.npy", batch_x)
                            np.save(f"nan_batch_y_{batch}.npy", batch_y)
                            logger.error(f"异常批次数据已保存至 nan_batch_x_{batch}.npy")
                        except:
                            pass
                        self.model.stop_training = True
    
    # 添加回调
    e.add_callbacks([
        NaNMonitor(),
        tf.keras.callbacks.ReduceLROnPlateau(  # 动态降低学习率
            monitor='loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
        )
    ])
    e.run()