import os
import copy

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import activations


class DeepSoundBaseRNN:
    ''' Create a RNN. '''
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

    def fit(self, X, y):
        
        # # 新增：展平嵌套列表（若存在）
        # if len(X) == 1 and isinstance(X[0], (list, np.ndarray)):
        #     X = X[0]  # 从[ [sample1, ..., sample42] ] 变为 [sample1, ..., sample42]
        
        
        
        # 新增：打印原始X的长度和类型
        print("Original X length:", len(X))
        print("Original X[0] type:", type(X[0]))
        print("Original X[0] length:", len(X[0]) if isinstance(X[0], (list, np.ndarray)) else "N/A")
        
        
        ''' Train network based on given data. '''
        self.classes_ = list(set(np.concatenate(y)))
        self.padding_class = len(self.classes_)

        X_pad = []
        
        
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))
            
            
        # 新增：打印原始X[0]的长度和类型
        print("X_pad length:", len(X_pad))
            
            

        y = keras.preprocessing.sequence.pad_sequences(
            y,
            padding='post',
            value=self.padding_class,
            dtype=object)

        X = np.asarray(X_pad).astype('float32')
        y = np.asarray(y).astype('float32')

        
        
        # 在 deep_sound.py 的 fit 方法中，X = np.asarray(X_pad).astype('float32') 之后
        print("After padding and converting to array:")
        print(f"X has NaN: {np.isnan(X).any()}")
        print(f"X has Inf: {np.isinf(X).any()}")
        
        
        # 添加打印语句验证形状
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        
        
        
        
        model_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=50)
        ]

        # Get sample weights if needed.
        sample_weights = None
        if self.set_sample_weights:
            sample_weights = self._get_samples_weights(y)

            
            
        # 新增！！！！        
        X = np.squeeze(X, axis=0)  # 去掉第0维的1，恢复成(42, 835, 1800) 
        print("X的内容：")
        print(X)
        print("X shape:", X.shape)
        
        
        # 在 X = np.squeeze(X, axis=0) 之后
        print("After squeezing:")
        print(f"X has NaN: {np.isnan(X).any()}")
        print(f"X has Inf: {np.isinf(X).any()}")

        
        # 在 X = np.squeeze(...) 之后，模型训练前
        print("X 数据统计：")
        print(f"最小值: {np.min(X)}")
        print(f"最大值: {np.max(X)}")
        print(f"是否包含 nan: {np.isnan(X).any()}")
        print(f"是否包含 inf: {np.isinf(X).any()}")

        # 若存在异常值，进行处理（示例：替换为0或截断）
        X = np.nan_to_num(X, nan=0.0, posinf=np.max(X[~np.isinf(X)]), neginf=np.min(X[~np.isinf(X)]))
        
        X = np.expand_dims(X, axis=-1)  # 补充通道维，形状变为(样本个数, seq_len, input_size, 1)
        print("X测试 shape after 补充通道维度:", X.shape)  # 新增打印
        
        # 新增：检查输入形状是否匹配模型要求
        input_shape = self.model.input_shape[1:]  # 模型预期输入形状（排除样本数维度）
        actual_shape = X.shape[1:]  # 实际输入形状（排除样本数维度）
        # if actual_shape != input_shape:
        #     logger.error(f"输入形状不匹配：模型预期{input_shape}，实际{actual_shape}")
        
        
        if self.feature_scaling:
            X = (X + 1.0) * 100
            print("缩放后 X训练 统计：", np.min(X), np.max(X))  # 关键检查

        
        
        self.model.fit(x=X,
                       y=y,
                       epochs=self.n_epochs,
                       verbose=1,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       # validation_split=0,
                       shuffle=True,
                       sample_weight=sample_weights,
                       callbacks=model_callbacks)

    def predict(self, X):
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        X = np.asarray(X_pad).astype('float32')
        
        # 在 deep_sound.py 的 predict 方法中，X = np.asarray(X_pad).astype('float32') 之后
        print("Predict: After padding and converting to array:")
        print(f"X has NaN: {np.isnan(X).any()}")
        print(f"X has Inf: {np.isinf(X).any()}")

                  
        print("X shape after padding:", X.shape)  # 新增打印  
        

        # X = X[0]
        # 新增！！！！        
        X = np.squeeze(X, axis=0)  # 去掉第0维的1，恢复成(42, 835, 1800) 
        print("X shape after padding:", X.shape)  # 新增打印
        print("X[的内容：")
        print(X)
        
        # 在 X = X[0] 之后
        print("Predict: After X = X[0]:")
        print(f"X has NaN: {np.isnan(X).any()}")
        print(f"X has Inf: {np.isinf(X).any()}")
        
        
        
        X = np.expand_dims(X, axis=-1)  # 补充通道维，形状变为(样本个数, seq_len, input_size, 1)
        print("X测试 shape after 补充通道维度:", X.shape)  # 新增打印
        
        if self.feature_scaling:
            X = (X + 1.0) * 100
            print("缩放后 X测试 统计：", np.min(X), np.max(X))  # 关键检查

                
        
        y_pred = self.model.predict(X).argmax(axis=-1)

        return y_pred

    def predict_proba(self, X):
        X_pad = []
        for i in X:
            X_pad.append(
                keras.preprocessing.sequence.pad_sequences(i,
                                                           padding='post',
                                                           value=-100.0,
                                                           dtype=float))

        X = np.asarray(X_pad).astype('float32')

        X = X[0]

        y_pred = self.model.predict(X)

        return y_pred

    def _get_samples_weights(self, y):
        # Get items counts.
        unique_classes, _, counts = np.unique(np.ravel(y),
        # _, _, counts = np.unique(np.ravel(y),
                                 return_counts=True,
                                 return_index=True,
                                 axis=0)

        # 新增：处理counts为0的情况
        counts = np.array(counts)
        if np.any(counts == 0):
            print("警告：存在样本数为0的类别！")
            counts = np.maximum(counts, 1)  # 避免除零
        
        
        # Get max without last element (padding class).
        class_weight = counts[:-1].max() / counts

        
        # ===== 添加检查语句：打印每个类别的样本数和权重 =====
        print("\n===== 填充置0前类别权重详情 =====")
        for cls, cnt, weight in zip(unique_classes, counts, class_weight):
            # 区分填充类别和普通类别
            cls_type = "填充类别" if cls == self.padding_class else "普通类别"
            print(f"类别 {cls} ({cls_type}): 样本数={cnt}, 权重={weight:.4f}")
        print("=======================\n")
        
        
        # Set padding class weight to zero.
        class_weight[self.padding_class] = 0.0
        
        
        # ===== 添加检查语句：打印每个类别的样本数和权重 =====
        print("\n===== 类别权重详情 =====")
        for cls, cnt, weight in zip(unique_classes, counts, class_weight):
            # 区分填充类别和普通类别
            cls_type = "填充类别" if cls == self.padding_class else "普通类别"
            print(f"类别 {cls} ({cls_type}): 样本数={cnt}, 权重={weight:.4f}")
        print("=======================\n")



        class_weight = {cls_num: weight for cls_num, weight in enumerate(class_weight)}
        sample_weight = np.zeros_like(y, dtype=float)

        print(class_weight)

        # Assign weight to every sample depending on class.
        for class_num, weight in class_weight.items():
            sample_weight[y == class_num] = weight

        return sample_weight

    def clear_params(self):
        self.model.set_weights(copy.deepcopy(self.weights_))


class DeepSound(DeepSoundBaseRNN):
    def __init__(self,
                 batch_size=5,
                 input_size=4000,
                 output_size=3,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True):
        ''' Create network instance of DeepSound arquitecture.
        '''
        super().__init__(batch_size,
                         n_epochs,
                         training_reshape,
                         set_sample_weights,
                         feature_scaling=feature_scaling)

        layers_config = [(32, 18, 3, activations.relu),
                         (32, 9, 1, activations.relu),
                         (128, 3, 1, activations.relu)]

        # Model definition
        cnn = Sequential()
        cnn.add(layers.Rescaling(scale=1.0))

        for ix_l, layer in enumerate(layers_config):
            for i in range(2):
                cnn.add(layers.Conv1D(layer[0],
                                      kernel_size=layer[1],
                                      strides=layer[2],
                                      activation=layer[3],
                                      padding=self.padding,
                                      data_format=self.data_format))
            if ix_l < (len(layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.2))

        cnn.add(layers.MaxPooling1D(4))
        cnn.add(layers.Flatten())
        cnn.add(layers.Dropout(rate=0.2))

        ffn = Sequential()

        ffn.add(layers.Dense(256, activation=activations.relu))
        ffn.add(layers.Dropout(rate=0.2))
        ffn.add(layers.Dense(128, activation=activations.relu))
        ffn.add(layers.Dropout(rate=0.2))
        ffn.add(layers.Dense(output_size, activation=activations.softmax))

        model = Sequential([layers.InputLayer(input_shape=(None, input_size, 1), name='input1'),
                            layers.TimeDistributed(cnn),
                            layers.Bidirectional(layers.GRU(128, activation="tanh",
                                                            return_sequences=True, dropout=0.2)),
                            layers.TimeDistributed(ffn)])

        model.compile(optimizer=Adagrad(),
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=['accuracy'])

        self.model = model
        self.weights_ = copy.deepcopy(model.get_weights())