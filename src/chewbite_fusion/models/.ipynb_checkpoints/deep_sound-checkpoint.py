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
            if hasattr(layer, 'layers'):  # 处理复合层
                for sub in layer.layers:
                    collect(sub, full_name)
            if isinstance(layer, layers.Bidirectional):  # 处理双向层内部
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
            print(f"极值: [{np.min(output):.6f}, {np.max(output):.6f}]")
            print(f"均值±标准差: {np.mean(output):.6f}±{np.std(output):.6f}")
            print(f"异常值: NaN={np.isnan(output).any()}, Inf={np.isinf(output).any()}")
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
        self.model_save_path = None  # 由子类初始化

    def fit(self, X, y):
        # 基础数据信息打印
        print(f"原始数据: 样本数={len(X)}, 首个样本类型={type(X[0])}")
        if isinstance(X[0], (list, np.ndarray)):
            print(f"首个样本长度={len(X[0])}")
        
        ''' 基于输入数据训练网络 '''
        self.classes_ = list(set(np.concatenate(y)))
        self.padding_class = len(self.classes_)
        print(f"类别信息: 共{len(self.classes_)}类, 填充类别索引={self.padding_class}")

        # 序列填充
        X_pad = [
            keras.preprocessing.sequence.pad_sequences(
                seq, padding='post', value=-100.0, dtype=float
            ) for seq in X
        ]
        print(f"填充后样本数={len(X_pad)}")
            
        # 标签处理
        y = keras.preprocessing.sequence.pad_sequences(
            y, padding='post', value=self.padding_class, dtype=object
        )

        # 转换为数组
        X = np.asarray(X_pad).astype('float32')
        y = np.asarray(y).astype('int32')  # 适配稀疏交叉熵

        # 基础数据校验
        print(f"数据形状: X={X.shape}, y={y.shape}")
        print(f"标签范围: [{y.min()}, {y.max()}], 含NaN={np.isnan(y).any()}")
        print(f"输入异常值: NaN={np.isnan(X).any()}, Inf={np.isinf(X).any()}")
        
        # 监控样本准备
        monitor_batch = X[:self.batch_size] if len(X) >= self.batch_size else X
        
        # 模型层信息打印
        print("\n===== 模型层结构 =====")
        all_layer_names = []
        def print_layer_names(layer, parent_name=""):
            full_name = f"{parent_name}/{layer.name}" if parent_name else layer.name
            all_layer_names.append(full_name)
            print(full_name)
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    print_layer_names(sub_layer, full_name)
            if isinstance(layer, layers.Bidirectional):
                print_layer_names(layer.layer, full_name)
        for layer in self.model.layers:
            print_layer_names(layer)
        print("======================\n")
        
        # 训练回调
        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            LayerOutputMonitor(
                model=self.model,
                layer_names=[
                    'time_distributed_cnn/cnn_subnetwork/conv1d_1',
                    'time_distributed_cnn/cnn_subnetwork/conv1d_2',
                    'time_distributed_cnn/cnn_subnetwork/conv1d_3',
                    'time_distributed_cnn/cnn_subnetwork/conv1d_4',
                    'time_distributed_cnn/cnn_subnetwork/conv1d_5',
                    'time_distributed_cnn/cnn_subnetwork/conv1d_6',
                    'time_distributed_cnn/cnn_subnetwork/max_pooling1d',
                    'bidirectional_gru',
                    'time_distributed_ffn/ffn_subnetwork/ffn_output'
                ],
                sample_batch=monitor_batch
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=20, min_lr=1e-8, verbose=1
            )
        ]

        # 样本权重计算
        sample_weights = self._get_samples_weights(y) if self.set_sample_weights else None

        # 维度调整与异常处理
        X = np.squeeze(X, axis=0)
        print(f"调整后X形状: {X.shape}")
        X = np.nan_to_num(
            X, 
            nan=0.0, 
            posinf=np.max(X[~np.isinf(X)]), 
            neginf=np.min(X[~np.isinf(X)])
        )

        # 形状匹配校验
        input_shape = self.model.input_shape[1:]
        actual_shape = X.shape[1:]
        if actual_shape != input_shape:
            raise ValueError(f"输入形状不匹配！模型预期{input_shape}，实际{actual_shape}")
        print(f"输入形状校验通过: 模型预期{input_shape}, 实际{actual_shape}")
        
        # 特征缩放
        if self.feature_scaling:
            X = (X + 1.0) * 100
            print(f"特征缩放后: 范围[{np.min(X):.2f}, {np.max(X):.2f}], 均值{np.mean(X):.2f}")
        
        # 模型训练
        self.model.fit(
            x=X,
            y=y,
            epochs=self.n_epochs,
            verbose=1,
            batch_size=self.batch_size,
            validation_split=0.2,
            shuffle=True,
            sample_weight=sample_weights,
            callbacks=model_callbacks
        )

    def predict(self, X):
        # 预测数据预处理
        X_pad = [
            keras.preprocessing.sequence.pad_sequences(
                seq, padding='post', value=-100.0, dtype=float
            ) for seq in X
        ]
        X = np.asarray(X_pad).astype('float32')
        
        # 数据校验
        print(f"预测输入形状: {X.shape}, 异常值: NaN={np.isnan(X).any()}, Inf={np.isinf(X).any()}")

        # 维度调整与异常处理
        X = np.squeeze(X, axis=0)
        print(f"预测调整后形状: {X.shape}")
        X = np.nan_to_num(
            X, 
            nan=0.0, 
            posinf=np.max(X[~np.isinf(X)]), 
            neginf=np.min(X[~np.isinf(X)])
        )
        
        # 特征缩放
        if self.feature_scaling:
            X = (X + 1.0) * 100
            print(f"预测特征缩放后: 范围[{np.min(X):.2f}, {np.max(X):.2f}]")
        
        # 模型预测
        y_pred = self.model.predict(X).argmax(axis=-1)
        return y_pred

    def predict_proba(self, X):
        # 概率预测预处理（与predict保持一致）
        X_pad = [
            keras.preprocessing.sequence.pad_sequences(
                seq, padding='post', value=-100.0, dtype=float
            ) for seq in X
        ]
        X = np.asarray(X_pad).astype('float32')
        X = np.squeeze(X, axis=0)
        
        if self.feature_scaling:
            X = (X + 1.0) * 100

        return self.model.predict(X)

    def _get_samples_weights(self, y):
        # 类别权重计算
        unique_classes, counts = np.unique(np.ravel(y), return_counts=True)
        
        # 处理零样本类别
        counts = np.maximum(counts, 1)  # 避免除零
        if np.any(counts == 1):
            zero_classes = unique_classes[counts == 1]
            print(f"警告：以下类别样本数为0，已自动调整: {zero_classes}")
        
        # 计算权重
        class_weight = counts[:-1].max() / counts  # 排除填充类别
        class_weight[self.padding_class] = 0.0  # 填充类别权重为0
        
        # 权重信息打印
        print("\n===== 类别权重详情 =====")
        for cls, cnt, weight in zip(unique_classes, counts, class_weight):
            cls_type = "填充类别" if cls == self.padding_class else "普通类别"
            print(f"类别 {cls} ({cls_type}): 样本数={cnt}, 权重={weight:.4f}")
        print("=======================\n")

        # 生成样本权重矩阵
        sample_weight = np.zeros_like(y, dtype=float)
        for cls_num, weight in enumerate(class_weight):
            sample_weight[y == cls_num] = weight
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
                 learning_rate=1e-6,  # 新增学习率参数
                 model_save_path="./model_checkpoints"):
        ''' DeepSound模型架构实现 '''
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)
        
        self.model_save_path = model_save_path
        self.time_steps = time_steps
        os.makedirs(self.model_save_path, exist_ok=True)  # 确保保存目录存在

        # CNN子网络（局部特征提取）
        layers_config = [
            (32, 18, 3, activations.relu),
            (32, 9, 1, activations.relu),
            (128, 3, 1, activations.relu)
        ]

        cnn = Sequential(name='cnn_subnetwork')
        cnn.add(layers.Rescaling(scale=1.0, name='input_rescaling'))  # 输入标准化基础

        for ix_l, (filters, kernel, stride, act) in enumerate(layers_config):
            # 双层卷积结构
            cnn.add(layers.Conv1D(
                filters, kernel, stride, activation=act,
                padding=self.padding, data_format=self.data_format,
                name=f'conv1d_{ix_l*2 + 1}'
            ))
            cnn.add(layers.Conv1D(
                filters, kernel, stride, activation=act,
                padding=self.padding, data_format=self.data_format,
                name=f'conv1d_{ix_l*2 + 2}'
            ))
            if ix_l < len(layers_config) - 1:
                cnn.add(layers.Dropout(0.2, name=f'dropout_{ix_l + 1}'))

        cnn.add(layers.MaxPooling1D(4, name='max_pooling1d'))
        cnn.add(layers.Flatten(name='flatten'))
        cnn.add(layers.Dropout(0.1, name='cnn_output_dropout'))

        # FFN子网络（分类头）
        ffn = Sequential(name='ffn_subnetwork')
        ffn.add(layers.Dense(256, activation=activations.relu, name='ffn_dense_1'))
        ffn.add(layers.Dropout(0.2, name='ffn_dropout_1'))
        ffn.add(layers.Dense(128, activation=activations.relu, name='ffn_dense_2'))
        ffn.add(layers.Dropout(0.2, name='ffn_dropout_2'))
        ffn.add(layers.Dense(output_size, activation=activations.softmax, name='ffn_output'))

        # 完整模型
        self.model = Sequential([
            layers.InputLayer(input_shape=(self.time_steps, input_size, 1), name='input1'),
            layers.TimeDistributed(cnn, name='time_distributed_cnn'),
            layers.Bidirectional(
                layers.GRU(128, activation="tanh", return_sequences=True, dropout=0.2),
                name='bidirectional_gru'
            ),
            layers.TimeDistributed(ffn, name='time_distributed_ffn')
        ])

        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['accuracy']
        )

        self.weights_ = copy.deepcopy(self.model.get_weights())
        print("模型初始化完成，结构如下：")
        self.model.summary()