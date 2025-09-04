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
        # 打印原始X的长度和类型（调试用）
        print("Original X length:", len(X))
        print("Original X[0] type:", type(X[0]))
        print("Original X[0] length:", len(X[0]) if isinstance(X[0], (list, np.ndarray)) else "N/A")
        
        ''' 基于输入数据训练网络 '''
        self.classes_ = list(set(np.concatenate(y)))
        self.padding_class = len(self.classes_)

        # 对输入序列进行填充，统一长度
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))
        print("X_pad length:", len(X_pad))  # 调试：打印填充后的样本数
            
        # 对标签序列进行填充
        y = keras.preprocessing.sequence.pad_sequences(
            y,
            padding='post',
            value=self.padding_class,
            dtype=object)

        # 转换为数组并指定数据类型
        X = np.asarray(X_pad).astype('float32')
        y = np.asarray(y).astype('int32')  # 标签用int32适配稀疏交叉熵

        # 检查标签有效性
        print("标签最小值:", y.min())
        print("标签最大值:", y.max())
        print("标签是否含NaN:", np.isnan(y).any())  # 检查标签NaN
        
        # 检查输入数据异常值
        print("After padding and converting to array:")
        print(f"X has NaN: {np.isnan(X).any()}")
        print(f"X has Inf: {np.isinf(X).any()}")
        
        # 打印数据形状
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        
        # 准备监控用的样本批次（取第一个批次）
        monitor_batch = X[:self.batch_size] if len(X) >= self.batch_size else X
        
        # 递归获取所有层的完整名称（修复复杂层结构的命名问题）
        print("\n===== 模型所有层完整名称 =====")
        all_layer_names = []
        def get_full_layer_names(layer, parent_name=""):
            full_name = f"{parent_name}/{layer.name}" if parent_name else layer.name
            all_layer_names.append(full_name)
            print(full_name)
            # 处理包含子层的结构（如Sequential、TimeDistributed）
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    get_full_layer_names(sub_layer, full_name)
            # 处理Bidirectional层的内部层
            if isinstance(layer, layers.Bidirectional):
                get_full_layer_names(layer.layer, full_name)
        for layer in self.model.layers:
            get_full_layer_names(layer)
        print("===========================\n")
        
        # 修正监控层名称（匹配实际层结构）
        monitor_layer_names = [
            'time_distributed_cnn/cnn_subnetwork/conv1d_1',
            'time_distributed_cnn/cnn_subnetwork/conv1d_2',
            'time_distributed_cnn/cnn_subnetwork/conv1d_3',
            'time_distributed_cnn/cnn_subnetwork/conv1d_4',
            'time_distributed_cnn/cnn_subnetwork/conv1d_5',
            'time_distributed_cnn/cnn_subnetwork/conv1d_6',
            'time_distributed_cnn/cnn_subnetwork/max_pooling1d',
            'bidirectional_gru',
            'time_distributed_ffn/ffn_subnetwork/ffn_output'
        ]
        
        # 定义训练回调
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
                layer_names=monitor_layer_names,
                sample_batch=monitor_batch
            ),
            tf.keras.callbacks.ReduceLROnPlateau(  # 学习率自动调整
                monitor='val_loss', factor=0.5, patience=20, min_lr=1e-8, verbose=1
            )
        ]

        # 计算样本权重（处理类别不平衡）
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

        # 调整输入维度（去除冗余的第0维）
        X = np.squeeze(X, axis=0)
        print("X shape after squeezing:", X.shape)
        
        # 再次检查异常值
        print("After squeezing:")
        print(f"X has NaN: {np.isnan(X).any()}")
        print(f"X has Inf: {np.isinf(X).any()}")

        # 处理异常值（替换NaN和Inf）
        X = np.nan_to_num(
            X, 
            nan=0.0, 
            posinf=np.max(X[~np.isinf(X)]), 
            neginf=np.min(X[~np.isinf(X)])
        )
        
        # 移除多余的通道维度添加（关键修复！）
        print("X shape after processing:", X.shape)
        
        # 检查输入形状与模型匹配性
        input_shape = self.model.input_shape[1:]
        actual_shape = X.shape[1:]
        print(f"模型预期输入形状: {input_shape}, 实际输入形状: {actual_shape}")
        if actual_shape != input_shape:
            raise ValueError(f"输入形状不匹配！模型预期{input_shape}，实际{actual_shape}")
        
        # 特征缩放（与预测保持一致）
        if self.feature_scaling:
            X = (X + 1.0) * 100
            print("缩放后 X训练 统计：", np.min(X), np.max(X), np.mean(X))
        
        # 数据统计信息（含异常值检查）
        print("X 数据统计（处理后）:")
        print(f"最小值: {np.min(X)}")
        print(f"最大值: {np.max(X)}")
        print(f"均值: {np.mean(X)}")
        print(f"标准差: {np.std(X)}")
        print(f"是否含NaN: {np.isnan(X).any()}")
        print(f"是否含Inf: {np.isinf(X).any()}")
        
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
        # 对输入进行填充处理（与训练保持一致）
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        X = np.asarray(X_pad).astype('float32')
        
        # 检查预测数据异常值
        print("Predict: After padding and converting to array:")
        print(f"X has NaN: {np.isnan(X).any()}")
        print(f"X has Inf: {np.isinf(X).any()}")
        print("X shape after padding:", X.shape)  

        # 调整维度（与训练处理一致）
        X = np.squeeze(X, axis=0)
        print("X shape after squeezing:", X.shape)
        
        # 检查异常值
        print("Predict: After squeezing:")
        print(f"X has NaN: {np.isnan(X).any()}")
        print(f"X has Inf: {np.isinf(X).any()}")
        
        # 移除多余的通道维度添加（关键修复！）
        print("X shape after processing:", X.shape)
        
        # 处理异常值
        X = np.nan_to_num(
            X, 
            nan=0.0, 
            posinf=np.max(X[~np.isinf(X)]), 
            neginf=np.min(X[~np.isinf(X)])
        )
        
        # 特征缩放（与训练保持一致）
        if self.feature_scaling:
            X = (X + 1.0) * 100
            print("缩放后 X测试 统计：", np.min(X), np.max(X), np.mean(X))
        
        # 模型预测
        y_pred = self.model.predict(X).argmax(axis=-1)
        return y_pred

    def predict_proba(self, X):
        # 预测概率（与predict逻辑保持一致的预处理）
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        X = np.asarray(X_pad).astype('float32')
        X = np.squeeze(X, axis=0)  # 移除冗余维度
        
        if self.feature_scaling:
            X = (X + 1.0) * 100

        y_pred = self.model.predict(X)
        return y_pred

    def _get_samples_weights(self, y):
        # 计算类别权重（处理类别不平衡）
        unique_classes, counts = np.unique(np.ravel(y), return_counts=True)
        
        # 处理样本数为0的类别
        counts = np.array(counts)
        if np.any(counts == 0):
            print("警告：存在样本数为0的类别！")
            counts = np.maximum(counts, 1)  # 避免除零错误
        
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

        # 转换为字典格式并生成样本权重矩阵
        class_weight_dict = {cls_num: weight for cls_num, weight in enumerate(class_weight)}
        sample_weight = np.zeros_like(y, dtype=float)
        for class_num, weight in class_weight_dict.items():
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
            (32, 18, 3, activations.relu),  # (filters, kernel_size, strides, activation)
            (32, 9, 1, activations.relu),
            (128, 3, 1, activations.relu)
        ]

        cnn = Sequential(name='cnn_subnetwork')
        cnn.add(layers.Rescaling(scale=1.0, name='input_rescaling'))  # 输入缩放层

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
            # 除最后一层外添加Dropout
            if ix_l < (len(layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.2, name=f'dropout_{ix_l + 1}'))

        cnn.add(layers.MaxPooling1D(4, name='max_pooling1d'))  # 降采样
        cnn.add(layers.Flatten(name='flatten'))  # 展平特征
        cnn.add(layers.Dropout(rate=0.1, name='cnn_output_dropout'))  # 输出Dropout

        # 定义FFN子网络（用于分类）
        ffn = Sequential(name='ffn_subnetwork')
        ffn.add(layers.Dense(256, activation=activations.relu, name='ffn_dense_1'))
        ffn.add(layers.Dropout(rate=0.2, name='ffn_dropout_1'))
        ffn.add(layers.Dense(128, activation=activations.relu, name='ffn_dense_2'))
        ffn.add(layers.Dropout(rate=0.2, name='ffn_dropout_2'))
        ffn.add(layers.Dense(output_size, activation=activations.softmax, name='ffn_output'))

        # 定义完整模型（时序特征提取+分类）
        # 输入形状：(time_steps, input_size, 1)
        model = Sequential([
            layers.InputLayer(input_shape=(self.time_steps, input_size, 1), name='input1'),
            layers.TimeDistributed(cnn, name='time_distributed_cnn'),  # 对每个时间步应用CNN
            layers.Bidirectional(
                layers.GRU(128, activation="tanh", return_sequences=True, dropout=0.2),
                name='bidirectional_gru'
            ),
            layers.TimeDistributed(ffn, name='time_distributed_ffn')  # 对每个时间步应用FFN
        ])

        # 编译模型（带梯度裁剪和学习率调整）
        model.compile(
            optimizer=Adam(learning_rate=1e-6, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['accuracy']
        )

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())  # 保存初始权重
        print("模型初始化完成，结构如下：")
        self.model.summary()