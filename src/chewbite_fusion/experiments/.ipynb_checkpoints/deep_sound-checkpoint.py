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


# 核心修改：将默认num_classes改为4（有效类别数，排除填充类别），并调整优化器参数
def get_model_instance(variable_params, num_classes=4):
    # 从参数网格中获取优化器参数，若未提供则使用默认
    optimizer_kwargs = variable_params.get('optimizer_kwargs', {
        'learning_rate': 1e-6,
        'clipnorm': 1.0,
        'clipvalue': 0.5
    })
    return DeepSound(
        input_size=1800,
        output_size=num_classes,  # 与有效类别数一致（0-3）
        n_epochs=2,
        batch_size=8,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True,
        optimizer_kwargs=optimizer_kwargs
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
        
        # 核心修改：明确有效类别数（排除填充类别4）
        n_classes = len(label_encoder.classes_)
        valid_classes = [cls for cls in unique_encoded_labels if cls != 4]  # 过滤填充类别
        valid_n_classes = len(np.unique(valid_classes)) if valid_classes else 0
        
        if np.any(unique_encoded_labels < 0) or np.any(unique_encoded_labels >= n_classes) and unique_encoded_labels[-1] != 4:
            raise ValueError(f"编码后标签超出范围[0, {n_classes-1}]，实际值: {unique_encoded_labels}")
        
        logger.info(f"标签编码成功，有效整数标签（不含填充）: {valid_classes}，有效类别数: {valid_n_classes}")
        logger.info(f"填充类别: 4（不参与模型训练）")
    except ValueError as e:
        logger.error(f"y标签异常: {str(e)}")
        raise
    
    # 核心修改：特征缩放改为标准化（更稳定）
    logger.info("应用特征标准化...")
    X_scaled = {}
    for key, segments in X_zavalla.items():
        seg_array = np.array(segments, dtype=np.float32)
        
        # 1. 更严格的异常值截断（2倍标准差）
        mean = np.mean(seg_array)
        std = np.std(seg_array)
        seg_array = np.clip(seg_array, mean - 2*std, mean + 2*std)  # 原3倍改为2倍
        # 2. 确保没有NaN/Inf（冗余检查）
        seg_array = np.nan_to_num(seg_array, nan=0.0, posinf=mean + 2*std, neginf=mean - 2*std)
        
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
    
    # 降低初始学习率（临时调试）
    optimizer_kwargs = {'learning_rate': 1e-4}  # 原学习率的1/10
    
    # 初始化实验：传入有效类别数（4）及优化器参数
    e = Experiment(
        get_model_instance,
        FeatureFactory_RawAudioData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deep_sound',
        manage_sequences=True,
        use_raw_data=True,
        data_augmentation=False,  # 先禁用增强，排查问题
        model_parameters_grid={'optimizer_kwargs': [optimizer_kwargs]}  # 传入优化器参数网格
    )
    
    # 训练过程中监控NaN（增强版）
    class NaNMonitor(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs is not None:
                for k, v in logs.items():
                    if np.isnan(v) or np.isinf(v):
                        logger.error(f"批次 {batch} 出现异常值: {k}={v}")
                        # 保存输入数据的统计信息及数据本身
                        try:
                            # 获取批次输入数据
                            batch_x = self.model.train_function.inputs[0].numpy()
                            batch_y = self.model.train_function.inputs[1].numpy()
                            
                            # 计算并记录输入数据范围统计
                            x_min = np.min(batch_x)
                            x_max = np.max(batch_x)
                            x_mean = np.mean(batch_x)
                            logger.error(f"批次X统计: min={x_min:.6f}, max={x_max:.6f}, mean={x_mean:.6f}")
                            
                            # 保存异常批次数据
                            np.save(f"nan_batch_x_{batch}.npy", batch_x)
                            np.save(f"nan_batch_y_{batch}.npy", batch_y)
                            logger.error(f"异常批次数据已保存至 nan_batch_x_{batch}.npy 和 nan_batch_y_{batch}.npy")
                        except Exception as e:
                            logger.error(f"保存异常批次失败: {str(e)}")  # 详细记录错误原因
                        self.model.stop_training = True  # 停止训练避免进一步错误
    
    # 添加WeightGradientChecker回调类
    class WeightGradientChecker(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            # 检查权重
            for layer in self.model.layers:
                if hasattr(layer, 'weights'):
                    for weight in layer.weights:
                        w_np = weight.numpy()
                        if np.isnan(w_np).any():
                            logger.error(f"批次 {batch}，层 {layer.name} 的权重存在NaN！")
                        if np.isinf(w_np).any():
                            logger.error(f"批次 {batch}，层 {layer.name} 的权重存在无穷大！")
            
            # 检查梯度
            try:
                with tf.GradientTape() as tape:
                    # 获取当前批次数据
                    batch_x = self.model.train_function.inputs[0]
                    batch_y = self.model.train_function.inputs[1]
                    y_pred = self.model(batch_x)
                    loss = self.model.compiled_loss(batch_y, y_pred)
                grads = tape.gradient(loss, self.model.trainable_weights)
                for grad, var in zip(grads, self.model.trainable_weights):
                    if grad is not None:
                        g_np = grad.numpy()
                        if np.isnan(g_np).any():
                            logger.error(f"批次 {batch}，变量 {var.name} 的梯度存在NaN！")
                        if np.isinf(g_np).any():
                            logger.error(f"批次 {batch}，变量 {var.name} 的梯度存在无穷大！")
            except Exception as e:
                logger.warning(f"梯度检查失败: {str(e)}")
    
    # 添加回调
    e.add_callbacks([
        NaNMonitor(),
        WeightGradientChecker(),  # 新增梯度和权重检查
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1  # 耐心从3减为2
        ),
        tf.keras.callbacks.LearningRateScheduler(  # 新增： epoch 10后强制降学习率
            lambda epoch: 1e-4 if epoch < 10 else 1e-5 if epoch < 30 else 1e-6
        )
    ])
    e.run()