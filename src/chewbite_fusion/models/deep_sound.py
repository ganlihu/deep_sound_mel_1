import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations


class LayerOutputMonitor(keras.callbacks.Callback):
    """自定义回调函数，监控模型中间层输出的数值状态（含NaN/Inf检查）"""
    def __init__(self, model, layer_names, sample_batch=None):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        # 创建获取中间层输出的模型（仅使用存在的层）
        self.valid_layers = [name for name in layer_names if name in self._get_all_layer_names()]
        self.invalid_layers = [name for name in layer_names if name not in self._get_all_layer_names()]
        if self.invalid_layers:
            print(f"警告：以下监控层不存在，已自动过滤：{self.invalid_layers}")
        self.feature_extractor = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(name).output for name in self.valid_layers]
        )
        self.sample_batch = sample_batch  # 用于监控的样本批次

    def _get_all_layer_names(self):
        """获取模型所有层的完整名称（支持嵌套结构）"""
        all_names = []
        def collect(layer, parent_name=""):
            full_name = f"{parent_name}/{layer.name}" if parent_name else layer.name
            all_names.append(full_name)
            # 处理包含子层的复合层
            if hasattr(layer, 'layers'):
                for sub in layer.layers:
                    collect(sub, full_name)
            # 处理双向层内部的循环层
            if isinstance(layer, layers.Bidirectional):
                collect(layer.layer, full_name)
        for layer in self.model.layers:
            collect(layer)
        return all_names

    def on_epoch_end(self, epoch, logs=None):
        if self.sample_batch is None or not self.valid_layers:
            return
        
        # 获取中间层输出并检查异常值
        layer_outputs = self.feature_extractor.predict(self.sample_batch, verbose=0)
        for name, output in zip(self.valid_layers, layer_outputs):
            print(f"\n===== Epoch {epoch} 层 {name} 输出监控 =====")
            print(f"形状: {output.shape}")
            print(f"最小值: {np.min(output):.6f}")
            print(f"最大值: {np.max(output):.6f}")
            print(f"均值: {np.mean(output):.6f}")
            print(f"标准差: {np.std(output):.6f}")
            print(f"含NaN: {np.isnan(output).any()}")
            print(f"含Inf: {np.isinf(output).any()}")
            print("=========================================\n")


class DeepSoundBaseRNN:
    ''' RNN基础类，提供训练、预测和权重管理功能 '''
    def __init__(self,
                 batch_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        self.classes_ = None
        self.padding_class = None

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.ghost_dim = 2
        self.padding = "valid"
        self.training_shape = training_reshape
        self.set_sample_weights = set_sample_weights
        self.feature_scaling = feature_scaling
        self.model = None  # 模型实例将在子类中初始化
        self.weights_ = None  # 用于存储初始权重

    def fit(self, X, y):
        # 1. 打印原始数据信息（关键：确认输入有42个样本）
        print("="*60)
        print("【原始数据信息】")
        print("Original X length:", len(X))
        print("Original X[0] type:", type(X[0]))
        print("Original X[0] length:", len(X[0]) if isinstance(X[0], (list, np.ndarray)) else "N/A")
        print("="*60)
        
        ''' 基于输入数据训练网络 '''
        self.classes_ = list(set(np.concatenate(y)))
        self.padding_class = len(self.classes_)

        # 2. 对输入序列进行填充，统一长度
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))
        print(f"填充后样本数: {len(X_pad)}")  # 这里应该是1（因为原始X是[list(42个样本)]）
            
        # 3. 对标签序列进行填充
        y = keras.preprocessing.sequence.pad_sequences(
            y,
            padding='post',
            value=self.padding_class,
            dtype=object)

        # 4. 转换为数组并指定数据类型
        X = np.asarray(X_pad).astype('float32')
        y = np.asarray(y).astype('int32')  # 标签用int32适配稀疏交叉熵

        # 5. 核心修复：正确处理维度，保留42个样本
        print(f"\n转换后X形状: {X.shape}")  # 此时是(1, 42, 835, 1800, 1)
        if X.shape[0] == 1:
            X = np.squeeze(X, axis=0)  # 去除第一个冗余维度，得到(42, 835, 1800, 1)
        print(f"去除冗余维度后X形状: {X.shape}")  # 预期：(42, 835, 1800, 1)
        print(f"当前样本数: {X.shape[0]}")  # 关键：确认是42个样本

        # 6. 检查数据有效性
        print(f"\n标签形状: {y.shape}")  # 预期：(42, 835)
        print("标签最小值:", y.min())
        print("标签最大值:", y.max())
        print("标签是否含NaN:", np.isnan(y).any())
        print(f"X含NaN: {np.isnan(X).any()}, X含Inf: {np.isinf(X).any()}")
        
        # 7. 准备监控用的样本批次（取前5个样本，而非只取1个）
        monitor_batch = X[:min(self.batch_size, X.shape[0])]  # 取前5个样本
        print(f"监控批次形状: {monitor_batch.shape}")  # 预期：(5, 835, 1800, 1)
        
        # 8. 递归获取所有层的完整名称（修复监控层名称错误）
        print("\n===== 模型所有层完整名称 =====")
        all_layer_names = []
        def get_full_layer_names(layer, parent_name=""):
            full_name = f"{parent_name}/{layer.name}" if parent_name else layer.name
            all_layer_names.append(full_name)
            print(full_name)
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    get_full_layer_names(sub_layer, full_name)
            if isinstance(layer, layers.Bidirectional):
                get_full_layer_names(layer.layer, full_name)
        for layer in self.model.layers:
            get_full_layer_names(layer)
        print("===========================\n")
        
        # 9. 修正监控层名称（匹配实际模型结构，从上面打印的名称中复制）
        monitor_layer_names = [
            'time_distributed_cnn/cnn_subnetwork/conv1d_1',
            'time_distributed_cnn/cnn_subnetwork/conv1d_2',
            'time_distributed_cnn/cnn_subnetwork/max_pooling1d',
            'bidirectional_gru',
            'time_distributed_ffn/ffn_subnetwork/ffn_output'
        ]
        
        # 10. 动态调整验证集（样本数≥5时才用0.2，避免报错）
        use_validation = X.shape[0] >= 5
        validation_split = 0.2 if use_validation else 0.0
        monitor_loss = 'val_loss' if use_validation else 'loss'
        monitor_acc = 'val_accuracy' if use_validation else 'accuracy'
        print(f"是否使用验证集: {use_validation}, 验证集比例: {validation_split}")
        
        # 11. 定义训练回调
        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True, monitor=monitor_loss),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, "best_model.h5"),
                monitor=monitor_acc,
                save_best_only=True,
                verbose=1
            ),
            LayerOutputMonitor(
                model=self.model,
                layer_names=monitor_layer_names,
                sample_batch=monitor_batch
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_loss, factor=0.5, patience=20, min_lr=1e-8, verbose=1
            )
        ]

        # 12. 计算样本权重（处理类别不平衡）
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        # 13. 处理异常值
        X = np.nan_to_num(
            X, 
            nan=0.0, 
            posinf=np.max(X[~np.isinf(X)]) if np.any(~np.isinf(X)) else 0.0,
            neginf=np.min(X[~np.isinf(X)]) if np.any(~np.isinf(X)) else 0.0
        )
        
        # 14. 特征缩放
        if self.feature_scaling:
            X = (X + 1.0) * 100
            print(f"\n缩放后X统计: min={np.min(X):.2f}, max={np.max(X):.2f}, mean={np.mean(X):.2f}")
        
        # 15. 检查输入形状与模型匹配性
        input_shape = self.model.input_shape[1:]  # 模型输入：(835, 1800, 1)
        actual_shape = X.shape[1:]
        print(f"\n模型预期输入形状: {input_shape}")
        print(f"实际输入形状: {actual_shape}")
        if actual_shape != input_shape:
            raise ValueError(f"输入形状不匹配！预期{input_shape}，实际{actual_shape}")
        
        # 16. 模型训练（使用42个样本，而非1个）
        print(f"\n【开始训练】样本数: {X.shape[0]}, 批次大小: {self.batch_size}")
        self.model.fit(
            x=X,  # 形状：(42, 835, 1800, 1)，42个样本
            y=y,  # 形状：(42, 835)，与样本数匹配
            epochs=self.n_epochs,
            verbose=1,
            batch_size=self.batch_size,
            validation_split=validation_split,  # 42个样本可以划分0.2验证集（8个）
            shuffle=True,
            sample_weight=sample_weights,
            callbacks=model_callbacks
        )

    def predict(self, X):
        # 对输入进行填充处理（与训练保持一致）
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        X = np.asarray(X_pad).astype('float32')
        
        # 处理维度（与训练一致）
        if X.shape[0] == 1:
            X = np.squeeze(X, axis=0)
        print(f"预测-处理后X形状: {X.shape}, 样本数: {X.shape[0]}")
        
        # 检查异常值
        print(f"Predict: X has NaN: {np.isnan(X).any()}, X has Inf: {np.isinf(X).any()}")
        
        # 处理异常值
        X = np.nan_to_num(
            X, 
            nan=0.0, 
            posinf=np.max(X[~np.isinf(X)]) if np.any(~np.isinf(X)) else 0.0,
            neginf=np.min(X[~np.isinf(X)]) if np.any(~np.isinf(X)) else 0.0
        )
        
        # 特征缩放
        if self.feature_scaling:
            X = (X + 1.0) * 100
            print("Predict: 缩放后X统计：", np.min(X), np.max(X), np.mean(X))
        
        # 模型预测
        y_pred = self.model.predict(X, verbose=0).argmax(axis=-1)
        return y_pred

    def predict_proba(self, X):
        # 预测概率（与predict逻辑保持一致）
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        X = np.asarray(X_pad).astype('float32')
        if X.shape[0] == 1:
            X = np.squeeze(X, axis=0)
        
        if self.feature_scaling:
            X = (X + 1.0) * 100

        y_pred = self.model.predict(X, verbose=0)
        return y_pred

    def _get_samples_weights(self, y):
        # 计算类别权重（处理类别不平衡）
        unique_classes, counts = np.unique(np.ravel(y), return_counts=True)
        
        # 处理样本数为0的类别
        counts = np.array(counts)
        if np.any(counts == 0):
            print("警告：存在样本数为0的类别！")
            counts = np.maximum(counts, 1)
        
        # 计算权重（基于反比频率）
        class_weight = counts[:-1].max() / counts  # 排除填充类别
        
        # 打印权重详情（填充前）
        print("\n===== 填充置0前类别权重详情 =====")
        for cls, cnt, weight in zip(unique_classes, counts, class_weight):
            cls_type = "填充类别" if cls == self.padding_class else "普通类别"
            print(f"类别 {cls} ({cls_type}): 样本数={cnt}, 权重={weight:.4f}")
        print("=======================\n")
        
        # 填充类别权重设为0（不参与训练）
        class_weight[self.padding_class] = 0.0
        
        # 打印权重详情（填充后）
        print("\n===== 类别权重详情 =====")
        for cls, cnt, weight in zip(unique_classes, counts, class_weight):
            cls_type = "填充类别" if cls == self.padding_class else "普通类别"
            print(f"类别 {cls} ({cls_type}): 样本数={cnt}, 权重={weight:.4f}")
        print("=======================\n")

        # 生成样本权重矩阵
        sample_weight = np.zeros_like(y, dtype=float)
        for class_num, weight in enumerate(class_weight):
            sample_weight[y == class_num] = weight

        return sample_weight

    def clear_params(self):
        """重置模型参数到初始状态"""
        if self.weights_ is not None:
            self.model.set_weights(copy.deepcopy(self.weights_))
        else:
            print("警告：未初始化模型权重，无法重置")


class DeepSound(DeepSoundBaseRNN):
    def __init__(self,
                 batch_size=5,
                 input_size=1800,  # 特征维度
                 time_steps=835,   # 时间步维度
                 output_size=4,    # 输出类别数
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True,
                 model_save_path="./model_checkpoints"):
        ''' DeepSound模型架构实现 '''
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)
        
        self.model_save_path = model_save_path
        self.time_steps = time_steps
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # 定义CNN子网络（用于局部特征提取）
        layers_config = [
            (32, 18, 3, activations.relu),
            (32, 9, 1, activations.relu),
            (128, 3, 1, activations.relu)
        ]

        cnn = Sequential(name='cnn_subnetwork')
        cnn.add(layers.Rescaling(scale=1.0, name='input_rescaling'))

        for ix_l, layer in enumerate(layers_config):
            # 第一组卷积
            cnn.add(layers.Conv1D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=layer[3],
                padding=self.padding,
                data_format=self.data_format,
                name=f'conv1d_{ix_l*2 + 1}'
            ))
            # 第二组卷积
            cnn.add(layers.Conv1D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=layer[3],
                padding=self.padding,
                data_format=self.data_format,
                name=f'conv1d_{ix_l*2 + 2}'
            ))
            if ix_l < (len(layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.2, name=f'dropout_{ix_l + 1}'))

        cnn.add(layers.MaxPooling1D(4, name='max_pooling1d'))
        cnn.add(layers.Flatten(name='flatten'))
        cnn.add(layers.Dropout(rate=0.1, name='cnn_output_dropout'))

        # 定义FFN子网络（用于分类）
        ffn = Sequential(name='ffn_subnetwork')
        ffn.add(layers.Dense(256, activation=activations.relu, name='ffn_dense_1'))
        ffn.add(layers.Dropout(rate=0.2, name='ffn_dropout_1'))
        ffn.add(layers.Dense(128, activation=activations.relu, name='ffn_dense_2'))
        ffn.add(layers.Dropout(rate=0.2, name='ffn_dropout_2'))
        ffn.add(layers.Dense(output_size, activation=activations.softmax, name='ffn_output'))

        # 定义完整模型
        model = Sequential([
            layers.InputLayer(input_shape=(self.time_steps, input_size, 1), name='input1'),
            layers.TimeDistributed(cnn, name='time_distributed_cnn'),
            layers.Bidirectional(
                layers.GRU(128, activation="tanh", return_sequences=True, dropout=0.2),
                name='bidirectional_gru'
            ),
            layers.TimeDistributed(ffn, name='time_distributed_ffn')
        ])

        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=1e-6, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['accuracy']
        )

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())
        print("模型初始化完成，结构如下：")
        self.model.summary()