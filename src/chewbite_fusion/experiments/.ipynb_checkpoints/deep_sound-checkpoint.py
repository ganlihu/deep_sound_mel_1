import logging
import tensorflow as tf
import numpy as np
from chewbite_fusion.models.deep_sound import DeepSound
from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features.feature_factories import FeatureFactory_RawAudioData
from chewbite_fusion.data.utils import NaNDetector

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    # 进一步降低学习率并增加梯度裁剪强度
    return DeepSound(
        input_size=1800,
        output_size=4,
        n_epochs=1,
        batch_size=2,  # 保持小批次
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True,
        # 新增：传递自定义优化器参数（在DeepSound类中需兼容）
        optimizer_kwargs={
            'learning_rate': 5e-5,  # 从1e-4降至5e-5
            'clipnorm': 2.0,        # 增强梯度裁剪
            'clipvalue': 1.0
        }
    )


@experiment()
def deep_sound():
    """ Experiment with Deep Sound architecture. """
    # 启用数值检查
    tf.debugging.enable_check_numerics()
    
    # 设置全局数值精度（避免浮点溢出）
    tf.keras.backend.set_floatx('float32')
    
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(
        window_width=window_width,
        window_overlap=window_overlap,
        include_movement_magnitudes=False,
        audio_sampling_frequency=6000
        # invalidate_cache=True
    )
    
    # 数据校验与清洗
    nan_detector = NaNDetector(verbose=True)
    X_zavalla = X['zavalla2022']
    y_zavalla = y['zavalla2022']
    
    # 检查并清理X数据
    try:
        for key, segments in X_zavalla.items():
            nan_detector.check_nan(segments, f"X片段 {key}")
        logger.info("X数据中未检测到NaN值")
    except ValueError as e:
        logger.error(f"X数据异常: {str(e)}")
        raise
    
    # 检查标签有效性（确保标签在合理范围）
    try:
        all_labels = []
        for key, labels in y_zavalla.items():
            nan_detector.check_nan(labels, f"y标签 {key}")
            all_labels.extend(labels)
        # 检查标签是否为整数且在[0, output_size-1]范围内
        unique_labels = np.unique(all_labels)
        if not np.issubdtype(unique_labels.dtype, np.integer):
            raise ValueError(f"标签必须为整数，实际类型: {unique_labels.dtype}")
        if np.any(unique_labels < 0) or np.any(unique_labels >= 4):  # output_size=4
            raise ValueError(f"标签超出范围[0,3]，实际值: {unique_labels}")
        logger.info(f"标签检查通过，有效标签: {unique_labels}")
    except ValueError as e:
        logger.error(f"y标签异常: {str(e)}")
        raise
    
    # 特征缩放增强（确保数值范围稳定）
    logger.info("应用特征缩放...")
    X_scaled = {}
    for key, segments in X_zavalla.items():
        # 转换为数组并归一化到[-1, 1]
        seg_array = np.array(segments, dtype=np.float32)
        # 避免除以0（添加微小epsilon）
        seg_max = np.max(np.abs(seg_array)) + 1e-8
        seg_scaled = seg_array / seg_max
        X_scaled[key] = seg_scaled
        # 检查缩放后范围
        logger.info(f"片段 {key} 缩放后范围: [{np.min(seg_scaled):.4f}, {np.max(seg_scaled):.4f}]")
    X['zavalla2022'] = X_scaled
    
    # 日志：数据基本信息
    logger.info("生成的片段数量: %s", len(X_zavalla.keys()))
    logger.info("片段编号示例: %s", list(X_zavalla.keys())[:5])
    
    # 初始化实验（禁用数据增强以排除干扰）
    e = Experiment(
        get_model_instance,
        FeatureFactory_RawAudioData,
        X, y,
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
    
    # 传递自定义回调给模型
    e.add_callbacks([NaNMonitor()])
    e.run()