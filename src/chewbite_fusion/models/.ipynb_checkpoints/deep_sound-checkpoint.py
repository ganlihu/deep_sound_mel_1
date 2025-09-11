import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class LayerOutputMonitor(keras.callbacks.Callback):
    """监控模型中间层输出，检测NaN/Inf等异常值"""
    def __init__(self, model, layer_names, sample_batch=None):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.valid_layers = [name for name in layer_names if name in self._get_all_layer_names()]
        self.invalid_layers = [name for name in layer_names if name not in self._get_all_layer_names()]
        if self.invalid_layers:
            print(f"警告：以下监控层不存在，已自动过滤：{self.invalid_layers}")
        self.feature_extractor = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(name).output for name in self.valid_layers]
        )
        self.sample_batch = sample_batch

    def _get_all_layer_names(self):
        all_names = []
        def collect(layer, parent_name=""):
            full_name = f"{parent_name}/{layer.name}" if parent_name else layer.name
            all_names.append(full_name)
            if hasattr(layer, 'layers'):
                for sub in layer.layers:
                    collect(sub, full_name)
            if isinstance(layer, layers.Bidirectional):
                collect(layer.layer, full_name)
        for layer in self.model.layers:
            collect(layer)
        return all_names

    def on_epoch_end(self, epoch, logs=None):
        if self.sample_batch is None or not self.valid_layers:
            return
        
        layer_outputs = self.feature_extractor.predict(self.sample_batch, verbose=0)
        for name, output in zip(self.valid_layers, layer_outputs):
            print(f"\n===== Epoch {epoch} 层 {name} 输出监控 =====")
            print(f"形状: {output.shape}")
            if np.isnan(output).any():
                print("最小值: NaN")
                print("最大值: NaN")
                print("均值: NaN")
                print("标准差: NaN")
            else:
                print(f"最小值: {np.min(output):.6f}")
                print(f"最大值: {np.max(output):.6f}")
                print(f"均值: {np.mean(output):.6f}")
                print(f"标准差: {np.std(output):.6f}")
            print(f"含NaN: {np.isnan(output).any()}")
            print(f"含Inf: {np.isinf(output).any()}")
            print("=========================================\n")


class GradientMonitor(keras.callbacks.Callback):
    """监控梯度范数，防止梯度爆炸"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # 每10个批次检查一次
            try:
                grads = self.model.optimizer.get_gradients(logs['loss'], self.model.trainable_weights)
                grad_norms = [tf.norm(g).numpy() if g is not None else 0.0 for g in grads]
                print(f"Batch {batch} 梯度范数最大值: {np.max(grad_norms):.4f}")
            except Exception as e:
                print(f"梯度监控出错: {str(e)}")


class DeepSoundBaseRNN:
    """RNN基础类，支持动态以最长序列长度填充"""
    def __init__(self,
                 batch_size=8,
                 n_epochs=1400,
                 input_size=1800,  # 特征维度（默认1800，可由子类覆盖）
                 set_sample_weights=True,
                 feature_scaling=True):
        self.classes_ = None
        self.padding_class = None
        self.max_seq_len = None  # 动态存储当前批次的最大序列长度
        self.input_size = input_size  # 特征维度（由子类传入或默认）

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.padding = "same"  # 卷积层用same填充
        self.set_sample_weights = set_sample_weights
        self.feature_scaling = feature_scaling
        self.model = None  # 模型将在fit中动态构建
        self.weights_ = None
        self.model_save_path = "./model_checkpoints"

    def _build_model(self, max_seq_len, output_size=4):
        """根据动态最大序列长度构建模型"""
        # 定义CNN子网络
        layers_config = [
            (32, 18, 3, activations.relu),
            (32, 9, 1, activations.relu),
            (128, 3, 1, activations.relu)
        ]

        cnn = Sequential(name='cnn_subnetwork')
        cnn.add(layers.Rescaling(scale=1.0, name='input_rescaling'))

        for ix_l, layer in enumerate(layers_config):
            # 卷积+BN+激活
            cnn.add(layers.Conv1D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=None,
                padding=self.padding,
                data_format=self.data_format,
                kernel_initializer=HeUniform(),
                name=f'conv1d_{ix_l*2 + 1}'
            ))
            cnn.add(layers.BatchNormalization(name=f'bn_{ix_l*2 + 1}'))
            cnn.add(layers.Activation(layer[3], name=f'act_{ix_l*2 + 1}'))

            cnn.add(layers.Conv1D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=None,
                padding=self.padding,
                data_format=self.data_format,
                kernel_initializer=HeUniform(),
                name=f'conv1d_{ix_l*2 + 2}'
            ))
            cnn.add(layers.BatchNormalization(name=f'bn_{ix_l*2 + 2}'))
            cnn.add(layers.Activation(layer[3], name=f'act_{ix_l*2 + 2}'))

            if ix_l < (len(layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.2, name=f'dropout_{ix_l + 1}'))

        cnn.add(layers.MaxPooling1D(4, name='max_pooling1d'))
        cnn.add(layers.Flatten(name='flatten'))
        cnn.add(layers.Dropout(rate=0.1, name='cnn_output_dropout'))

        # 定义FFN子网络
        ffn = Sequential(name='ffn_subnetwork')
        ffn.add(layers.Dense(256, activation=None, kernel_initializer=HeUniform(), name='ffn_dense_1'))
        ffn.add(layers.BatchNormalization(name='ffn_bn_1'))
        ffn.add(layers.Activation(activations.relu, name='ffn_act_1'))
        ffn.add(layers.Dropout(rate=0.2, name='ffn_dropout_1'))
        
        ffn.add(layers.Dense(128, activation=None, kernel_initializer=HeUniform(), name='ffn_dense_2'))
        ffn.add(layers.BatchNormalization(name='ffn_bn_2'))
        ffn.add(layers.Activation(activations.relu, name='ffn_act_2'))
        ffn.add(layers.Dropout(rate=0.2, name='ffn_dropout_2'))
        
        ffn.add(layers.Dense(output_size, activation=activations.softmax, name='ffn_output'))

        # 定义完整模型（输入形状为动态max_seq_len）
        model = Sequential([
            layers.InputLayer(input_shape=(max_seq_len, self.input_size, 1), name='input1'),
            layers.TimeDistributed(cnn, name='time_distributed_cnn'),
            layers.Bidirectional(
                layers.GRU(128, 
                           activation="tanh", 
                           return_sequences=True, 
                           dropout=0.2,
                           recurrent_dropout=0.1,
                           kernel_initializer=HeUniform()),
                name='bidirectional_gru'
            ),
            layers.TimeDistributed(ffn, name='time_distributed_ffn')
        ])

        # 编译模型
        model.compile(
            optimizer=Adam(
                learning_rate=5e-7,
                clipnorm=1.0,
                clipvalue=0.5
            ),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['accuracy']
        )

        return model

    def fit(self, X, y):
        # 1. 数据预处理：提取嵌套样本并计算动态最大长度
        print("="*60)
        print("【训练数据信息】")
        print(f"原始X类型: {type(X)}, 长度: {len(X) if isinstance(X, (list, np.ndarray)) else 'N/A'}")
        print(f"原始y类型: {type(y)}, 长度: {len(y) if isinstance(y, (list, np.ndarray)) else 'N/A'}")
        print("="*60)
        
        # 提取嵌套的样本（如X是[ [样本1], [样本2], ... ]，提取为[样本1, 样本2, ...]）
        if isinstance(X, list) and len(X) == 1 and isinstance(X[0], (list, np.ndarray)):
            X = X[0]
        if isinstance(y, list) and len(y) == 1 and isinstance(y[0], (list, np.ndarray)):
            y = y[0]
        
        # --------------------------
        # 新增：将列表样本转换为NumPy数组
        print("\n===== 转换样本为NumPy数组 =====")
        X_array = []
        for i, sample in enumerate(X):
            if isinstance(sample, list):
                # 尝试将列表转换为数组
                try:
                    sample_array = np.array(sample, dtype='float32')
                    X_array.append(sample_array)
                    print(f"样本{i}：已从list转换为数组，形状={sample_array.shape}")
                except ValueError as e:
                    print(f"样本{i}：列表转换为数组失败！错误：{e}")
                    raise  # 终止程序，因为样本格式错误
            elif isinstance(sample, np.ndarray):
                X_array.append(sample)
                print(f"样本{i}：已是数组，形状={sample.shape}")
            else:
                raise TypeError(f"样本{i}：既不是list也不是数组，类型={type(sample)}")
        X = X_array  # 替换为转换后的数组列表
        print("===========================\n")
        # --------------------------
        
        # 打印所有样本的形状详情（关键调试信息）
        print("\n===== 所有样本形状详情 =====")
        for i, sample in enumerate(X):
            if isinstance(sample, np.ndarray):
                print(f"样本{i}：维度={sample.ndim}，形状={sample.shape}")
                if sample.ndim == 2:
                    print(f"  → 窗口数={sample.shape[0]}, 每个窗口特征维度={sample.shape[1]}")
                # 处理1维样本（添加特征维度）
                if sample.ndim == 1:
                    print(f"  → 检测到1维样本，自动扩展为2维 (窗口数, 1)")
            else:
                print(f"样本{i}：非数组类型={type(sample)}")
        print("===========================\n")
        
        # 将1维样本转换为2维（添加特征维度）
        X = [
            np.expand_dims(sample, axis=-1) if (isinstance(sample, np.ndarray) and sample.ndim == 1) 
            else sample 
            for sample in X
        ]
        
        # 动态计算当前批次的最大序列长度（窗口数）
        if isinstance(X, (list, np.ndarray)):
            self.max_seq_len = max(len(sample) for sample in X) if X else 0
            print(f"当前批次最长序列长度（窗口数）: {self.max_seq_len}")
        else:
            raise ValueError("X必须是列表或NumPy数组")

        # 2. 同步填充X和y到动态最大长度
        target_len = self.max_seq_len
        
        # 填充X（特征）前先统一特征维度
        X_padded = []
        for sample in X:
            # 确保样本是2维数组
            if not isinstance(sample, np.ndarray) or sample.ndim != 2:
                raise ValueError(f"样本必须是2维数组，实际样本形状: {sample.shape if isinstance(sample, np.ndarray) else type(sample)}")
            
            seq_len, feat_dim = sample.shape
            
            # 统一特征维度到input_size
            if feat_dim != self.input_size:
                print(f"样本特征维度不匹配: 实际{feat_dim}，预期{self.input_size}，自动调整")
                if feat_dim > self.input_size:
                    sample = sample[:, :self.input_size]  # 截断
                else:
                    # 补全（用0填充）
                    pad_width = ((0, 0), (0, self.input_size - feat_dim))
                    sample = np.pad(sample, pad_width, mode='constant', constant_values=0.0)
            
            # 填充时间步长到target_len
            padded = keras.preprocessing.sequence.pad_sequences(
                sample,
                maxlen=target_len,
                padding='post',
                value=-1.0,  # 临时填充值
                dtype='float32'
            )
            X_padded.append(padded)
        
        # 转换为数组（此时应形状统一）
        try:
            X = np.array(X_padded, dtype='float32')
        except ValueError as e:
            print(f"转换为数组失败: {e}")
            # 打印每个填充后样本的形状，定位问题
            print("填充后样本形状:")
            for i, p in enumerate(X_padded):
                print(f"样本{i}: {p.shape}")
            raise  # 重新抛出异常
        
        # 添加通道维度（模型需要(样本数, 时间步, 特征, 1)）
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)
        print(f"X填充后形状: {X.shape}")  # 预期：(样本数, max_seq_len, input_size, 1)

        # 填充y（标签）
        self.classes_ = list(set(np.concatenate(y))) if y and isinstance(y[0], (list, np.ndarray)) else []
        self.padding_class = max(self.classes_) + 1 if self.classes_ else 0  # 新增填充类别
        y_padded = keras.preprocessing.sequence.pad_sequences(
            y,
            maxlen=target_len,
            padding='post',
            value=self.padding_class,  # 用填充类别填充
            dtype='int32'
        )
        y = y_padded
        print(f"y填充后形状: {y.shape}")  # 预期：(样本数, max_seq_len)
        print(f"类别: {self.classes_}, 填充类别编号: {self.padding_class}")

        # 3. 处理X的填充值（替换为非填充区域的均值）
        non_pad_mask = X != -1.0
        if np.any(non_pad_mask):
            mean_val = np.mean(X[non_pad_mask])
            X[~non_pad_mask] = mean_val
            print(f"X填充值替换为均值: {mean_val:.4f}")

        # 4. 特征标准化
        if self.feature_scaling and np.any(non_pad_mask):
            mean = np.mean(X[non_pad_mask])
            std = np.std(X[non_pad_mask])
            X[non_pad_mask] = (X[non_pad_mask] - mean) / (std + 1e-8)
            print(f"标准化后X统计: min={np.min(X):.4f}, max={np.max(X):.4f}, mean={np.mean(X):.4f}")

        # 5. 动态构建模型（根据当前max_seq_len）
        output_size = len(self.classes_) + 1 if self.classes_ else 4  # 包含填充类别
        self.model = self._build_model(max_seq_len=self.max_seq_len, output_size=output_size)
        self.weights_ = copy.deepcopy(self.model.get_weights())
        print("\n模型初始化完成（动态输入形状），结构如下：")
        self.model.summary()

        # 6. 准备监控批次和回调
        monitor_batch = X[:min(self.batch_size, X.shape[0])]
        print(f"\n监控批次形状: {monitor_batch.shape}")

        # 动态验证集
        use_validation = X.shape[0] >= 5
        validation_split = 0.2 if use_validation else 0.0
        monitor_loss = 'val_loss' if use_validation else 'loss'
        monitor_acc = 'val_accuracy' if use_validation else 'accuracy'
        print(f"使用验证集: {use_validation}, 验证比例: {validation_split}")

        # 回调函数
        model_callbacks = [
            EarlyStopping(patience=50, restore_best_weights=True, monitor=monitor_loss),
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, "best_model.h5"),
                monitor=monitor_acc,
                save_best_only=True,
                verbose=1
            ),
            LayerOutputMonitor(
                model=self.model,
                layer_names=['time_distributed_cnn', 'bidirectional_gru', 'time_distributed_ffn'],
                sample_batch=monitor_batch
            ),
            ReduceLROnPlateau(
                monitor=monitor_loss, factor=0.5, patience=15, min_lr=1e-8, verbose=1
            ),
            GradientMonitor(model=self.model)
        ]

        # 7. 样本权重（填充类别权重设为0）
        sample_weights = None
        if self.set_sample_weights and y.size > 0:
            sample_weights = self._get_samples_weights(y)
            sample_weights = np.clip(sample_weights, 0.0, 10.0)
            print(f"样本权重范围: [{np.min(sample_weights):.4f}, {np.max(sample_weights):.4f}]")

        # 8. 模型训练
        print(f"\n【开始训练】样本数: {X.shape[0]}, 批次大小: {self.batch_size}")
        self.model.fit(
            x=X,
            y=y,
            epochs=self.n_epochs,
            verbose=1,
            batch_size=self.batch_size,
            validation_split=validation_split,
            shuffle=True,
            sample_weight=sample_weights,
            callbacks=model_callbacks
        )

    def predict(self, X):
        # 预测阶段：使用训练时的max_seq_len填充
        if self.max_seq_len is None:
            raise RuntimeError("请先调用fit方法训练模型，获取动态最大序列长度")
        
        # 提取嵌套样本
        if isinstance(X, list) and len(X) == 1 and isinstance(X[0], (list, np.ndarray)):
            X = X[0]
        
        # 转换列表样本为数组
        X_array = []
        for i, sample in enumerate(X):
            if isinstance(sample, list):
                try:
                    sample_array = np.array(sample, dtype='float32')
                    X_array.append(sample_array)
                except ValueError as e:
                    print(f"预测样本{i}：列表转换为数组失败！错误：{e}")
                    raise
            elif isinstance(sample, np.ndarray):
                X_array.append(sample)
            else:
                raise TypeError(f"预测样本{i}：既不是list也不是数组，类型={type(sample)}")
        X = X_array
        
        # 处理1维样本
        X = [
            np.expand_dims(sample, axis=-1) if (isinstance(sample, np.ndarray) and sample.ndim == 1) 
            else sample 
            for sample in X
        ]
        
        # 用训练时的max_seq_len填充
        X_padded = []
        for sample in X:
            # 统一特征维度
            if sample.ndim != 2:
                raise ValueError(f"预测样本必须是2维数组，实际形状: {sample.shape}")
            
            seq_len, feat_dim = sample.shape
            if feat_dim != self.input_size:
                if feat_dim > self.input_size:
                    sample = sample[:, :self.input_size]
                else:
                    pad_width = ((0, 0), (0, self.input_size - feat_dim))
                    sample = np.pad(sample, pad_width, mode='constant', constant_values=0.0)
            
            padded = keras.preprocessing.sequence.pad_sequences(
                sample,
                maxlen=self.max_seq_len,
                padding='post',
                value=-1.0,
                dtype='float32'
            )
            X_padded.append(padded)
        
        X = np.array(X_padded, dtype='float32')
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)
        
        # 替换填充值并标准化（与训练一致）
        non_pad_mask = X != -1.0
        if np.any(non_pad_mask):
            mean_val = np.mean(X[non_pad_mask])
            X[~non_pad_mask] = mean_val
            mean = np.mean(X[non_pad_mask])
            std = np.std(X[non_pad_mask])
            X[non_pad_mask] = (X[non_pad_mask] - mean) / (std + 1e-8)
        
        print(f"预测输入形状: {X.shape}")
        y_pred = self.model.predict(X, verbose=0).argmax(axis=-1)
        return y_pred

    def predict_proba(self, X):
        if self.max_seq_len is None:
            raise RuntimeError("请先调用fit方法训练模型")
        
        if isinstance(X, list) and len(X) == 1 and isinstance(X[0], (list, np.ndarray)):
            X = X[0]
        
        # 转换列表样本为数组
        X_array = []
        for i, sample in enumerate(X):
            if isinstance(sample, list):
                try:
                    sample_array = np.array(sample, dtype='float32')
                    X_array.append(sample_array)
                except ValueError as e:
                    print(f"预测样本{i}：列表转换为数组失败！错误：{e}")
                    raise
            elif isinstance(sample, np.ndarray):
                X_array.append(sample)
            else:
                raise TypeError(f"预测样本{i}：既不是list也不是数组，类型={type(sample)}")
        X = X_array
        
        # 处理1维样本
        X = [
            np.expand_dims(sample, axis=-1) if (isinstance(sample, np.ndarray) and sample.ndim == 1) 
            else sample 
            for sample in X
        ]
        
        X_padded = []
        for sample in X:
            # 统一特征维度
            if sample.ndim != 2:
                raise ValueError(f"预测样本必须是2维数组，实际形状: {sample.shape}")
            
            seq_len, feat_dim = sample.shape
            if feat_dim != self.input_size:
                if feat_dim > self.input_size:
                    sample = sample[:, :self.input_size]
                else:
                    pad_width = ((0, 0), (0, self.input_size - feat_dim))
                    sample = np.pad(sample, pad_width, mode='constant', constant_values=0.0)
            
            padded = keras.preprocessing.sequence.pad_sequences(
                sample,
                maxlen=self.max_seq_len,
                padding='post',
                value=-1.0,
                dtype='float32'
            )
            X_padded.append(padded)
        
        X = np.array(X_padded, dtype='float32')
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)
        
        non_pad_mask = X != -1.0
        if np.any(non_pad_mask):
            mean_val = np.mean(X[non_pad_mask])
            X[~non_pad_mask] = mean_val
            mean = np.mean(X[non_pad_mask])
            std = np.std(X[non_pad_mask])
            X[non_pad_mask] = (X[non_pad_mask] - mean) / (std + 1e-8)

        y_pred = self.model.predict(X, verbose=0)
        return y_pred

    def _get_samples_weights(self, y):
        # 计算样本权重，填充类别权重设为0
        unique_classes, counts = np.unique(np.ravel(y), return_counts=True)
        counts = np.maximum(counts, 1)
        
        # 对数平滑权重
        class_weight = np.log((counts.max() / counts) + 1.0)
        
        # 填充类别权重设为0
        if self.padding_class in unique_classes:
            pad_idx = np.where(unique_classes == self.padding_class)[0][0]
            class_weight[pad_idx] = 0.0
        
        # 打印权重信息
        print("\n===== 类别权重 =====")
        for cls, cnt, weight in zip(unique_classes, counts, class_weight):
            cls_type = "填充类别" if cls == self.padding_class else "普通类别"
            print(f"类别 {cls} ({cls_type}): 样本数={cnt}, 权重={weight:.4f}")
        print("====================\n")

        # 生成样本权重矩阵
        sample_weight = np.zeros_like(y, dtype=float)
        for class_num, weight in zip(unique_classes, class_weight):
            sample_weight[y == class_num] = weight

        return sample_weight

    def clear_params(self):
        if self.weights_ is not None and self.model is not None:
            self.model.set_weights(copy.deepcopy(self.weights_))
        else:
            print("警告：未初始化模型权重，无法重置")


# 核心修复：恢复DeepSound子类，保持与原有代码的兼容性
class DeepSound(DeepSoundBaseRNN):
    def __init__(self,
                 batch_size=5,
                 input_size=4000,  # 恢复原始默认值4000
                 output_size=3,     # 恢复原始默认值3
                 n_epochs=1400,
                 training_reshape=False,  # 原始参数，虽未使用但保留兼容
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' 保持与原始DeepSound类一致的接口 '''
        # 调用父类构造函数，传递必要参数
        super().__init__(
            batch_size=batch_size,
            n_epochs=n_epochs,
            input_size=input_size,
            set_sample_weights=set_sample_weights,
            feature_scaling=feature_scaling
        )
        # 保留原始参数，确保兼容性
        self.training_reshape = training_reshape
        self.output_size = output_size  # 可用于后续扩展


# 使用示例（与原始代码用法一致）
if __name__ == "__main__":
    # 模拟42个样本，混合列表和数组类型，测试转换功能
    n_samples = 42
    input_size = 4000
    max_seq = 835
    
    # 生成模拟数据（部分为列表，部分为数组）
    X = []
    y = []
    for i in range(n_samples):
        seq_len = np.random.randint(200, max_seq + 1)  # 随机窗口数
        # 一半样本为列表，一半为数组，测试转换功能
        if i % 2 == 0:
            # 列表类型样本
            X.append([[np.random.rand() for _ in range(input_size)] for _ in range(seq_len)])
        else:
            # 数组类型样本
            X.append(np.random.rand(seq_len, input_size))
        y.append(np.random.randint(0, 4, seq_len))     # 标签：(窗口数,)
    
    # 初始化模型
    model = DeepSound(
        batch_size=5,
        input_size=input_size,
        output_size=4,
        n_epochs=10
    )
    
    # 训练模型
    model.fit(X, y)
    
    # 预测（包含列表类型样本）
    test_samples = X[:5]  # 包含列表和数组类型
    y_pred = model.predict(test_samples)
    print(f"预测结果形状: {y_pred.shape}")
