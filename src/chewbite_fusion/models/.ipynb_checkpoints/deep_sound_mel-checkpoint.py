import os
import copy
import time
import numpy as np
import tensorflow as tf
import psutil
from datetime import datetime
from typing import List, Tuple, Optional, Union, Dict
import logging

# 初始化日志对象（名称需与项目保持一致）
logger = logging.getLogger('yaer')  # 与其他文件保持相同的logger名称
try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False
    print("警告：未安装pynvml库，无法监控GPU详情，请运行'pip install pynvml'安装")

# 设置GPU内存动态增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"已启用GPU内存动态增长，检测到{len(gpus)}个GPU")
    except RuntimeError as e:
        print(f"设置GPU内存增长失败: {e}")

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras import activations
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from chewbite_fusion.data.utils import NaNDetector
import traceback


class ResourceLogger:
    """资源日志记录器，用于收集和保存系统资源使用信息"""
    def __init__(self, log_file: Optional[str] = None):
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = f"training_resource_log_{timestamp}.txt"
        else:
            self.log_file = log_file
        
        self.log_entries: List[str] = []
        self.system_info: Dict = {}
        self.memory_peaks = {
            'cpu_memory_used': 0.0,
            'gpu_memory_used': {},
            'process_memory_used': 0.0
        }
        self.initialized = False
        self.gpu_handles: List = []
        
        if pynvml_available:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
                self.initialized = True
            except pynvml.NVMLError as e:
                self.log(f"GPU监控初始化失败: {e}")
    
    def log(self, message: str, include_timestamp: bool = True) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}" if include_timestamp else message
        self.log_entries.append(entry)
        print(message)
    
    def record_system_info(self) -> None:
        self.log("===== 系统基本信息 =====")
        
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        self.system_info['cpu'] = {
            'logical_cores': cpu_count_logical,
            'physical_cores': cpu_count_physical
        }
        self.log(f"CPU核心数: {cpu_count_logical} (物理核心: {cpu_count_physical})")
        
        mem = psutil.virtual_memory()
        total_mem_gb = mem.total / (1024**3)
        available_mem_gb = mem.available / (1024**3)
        self.system_info['memory'] = {
            'total_gb': total_mem_gb,
            'available_gb': available_mem_gb
        }
        self.log(f"总内存: {total_mem_gb:.2f} GB")
        self.log(f"初始可用内存: {available_mem_gb:.2f} GB")
        
        self.system_info['gpus'] = []
        if pynvml_available and self.initialized:
            for i, handle in enumerate(self.gpu_handles):
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
                total_vram_gb = info.total / (1024**3)
                available_vram_gb = info.free / (1024**3)
                
                gpu_info = {
                    'index': i,
                    'name': gpu_name,
                    'total_vram_gb': total_vram_gb,
                    'available_vram_gb': available_vram_gb
                }
                self.system_info['gpus'].append(gpu_info)
                self.log(f"GPU {i}: {gpu_name}")
                self.log(f"  总显存: {total_vram_gb:.2f} GB")
                self.log(f"  初始可用显存: {available_vram_gb:.2f} GB")
                self.memory_peaks['gpu_memory_used'][i] = 0.0
        
        self.log("========================\n")
    
    def update_memory_peaks(self, batch: Optional[int] = None, epoch: Optional[int] = None) -> None:
        mem = psutil.virtual_memory()
        current_used = mem.used / (1024**3)
        if current_used > self.memory_peaks['cpu_memory_used']:
            self.memory_peaks['cpu_memory_used'] = current_used
        
        process = psutil.Process(os.getpid())
        current_process_used = process.memory_info().rss / (1024**3)
        if current_process_used > self.memory_peaks['process_memory_used']:
            self.memory_peaks['process_memory_used'] = current_process_used
        
        if pynvml_available and self.initialized:
            for i, handle in enumerate(self.gpu_handles):
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_gpu_used = mem_info.used / (1024**3)
                    if current_gpu_used > self.memory_peaks['gpu_memory_used'][i]:
                        self.memory_peaks['gpu_memory_used'][i] = current_gpu_used
                except pynvml.NVMLError as e:
                    self.log(f"更新GPU {i} 内存峰值失败: {e}")
    
    def record_training_summary(self, training_info: Dict[str, str]) -> None:
        self.log("\n===== 训练总结 =====")
        for key, value in training_info.items():
            self.log(f"{key}: {value}")
        self.log("====================\n")
    
    def record_memory_summary(self) -> None:
        self.log("\n===== 内存使用总结 =====")
        
        mem = psutil.virtual_memory()
        total_mem_gb = mem.total / (1024**3)
        used_mem_gb = mem.used / (1024**3)
        available_mem_gb = mem.available / (1024**3)
        
        self.log(f"CPU内存状态:")
        self.log(f"  总内存: {total_mem_gb:.2f} GB")
        self.log(f"  已使用: {used_mem_gb:.2f} GB ({mem.percent}%)")
        self.log(f"  剩余可用: {available_mem_gb:.2f} GB")
        self.log(f"  使用峰值: {self.memory_peaks['cpu_memory_used']:.2f} GB")
        
        self.log(f"进程内存使用峰值: {self.memory_peaks['process_memory_used']:.2f} GB")
        
        if pynvml_available and self.initialized:
            for i, handle in enumerate(self.gpu_handles):
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used_vram_gb = mem_info.used / (1024**3)
                    total_vram_gb = mem_info.total / (1024**3)
                    available_vram_gb = mem_info.free / (1024**3)
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(gpu_name, bytes):
                        gpu_name = gpu_name.decode('utf-8')
                    
                    self.log(f"GPU {i} ({gpu_name}) 显存状态:")
                    self.log(f"  总显存: {total_vram_gb:.2f} GB")
                    self.log(f"  已使用: {used_vram_gb:.2f} GB ({mem_info.used/mem_info.total*100:.1f}%)")
                    self.log(f"  剩余可用: {available_vram_gb:.2f} GB")
                    self.log(f"  使用峰值: {self.memory_peaks['gpu_memory_used'][i]:.2f} GB")
                except pynvml.NVMLError as e:
                    self.log(f"获取GPU {i} 内存信息失败: {e}")
        
        self.log("=======================\n")
    
    def save_log(self) -> None:
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.log_entries))
            self.log(f"\n日志已保存到: {os.path.abspath(self.log_file)}", include_timestamp=False)
        except Exception as e:
            print(f"保存日志文件失败: {e}")
        
        if pynvml_available and self.initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class GPUUsageMonitor(keras.callbacks.Callback):
    """实时监控所有GPU和系统资源使用情况的回调"""
    def __init__(self, interval: int = 10, resource_logger: Optional[ResourceLogger] = None):
        super().__init__()
        self.interval = interval
        self.resource_logger = resource_logger or ResourceLogger()
        self.start_time = time.time()
        
    def on_train_begin(self, logs=None) -> None:
        self.start_time = time.time()
        self.resource_logger.log("\n===== 系统资源监控初始化 =====")
        self.resource_logger.record_system_info()
    
    def on_train_batch_end(self, batch: int, logs=None) -> None:
        if batch % self.interval == 0:
            self.resource_logger.update_memory_peaks(batch=batch)
            
    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self.resource_logger.update_memory_peaks(epoch=epoch)
        self.resource_logger.log(f"\n===== Epoch {epoch} 资源使用统计 =====")
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        self.resource_logger.log(f"CPU使用率: {cpu_percent}%")
        self.resource_logger.log(f"内存状态: "
                               f"已用 {mem.used / (1024**3):.2f} GB / "
                               f"总 {mem.total / (1024**3):.2f} GB / "
                               f"可用 {mem.available / (1024**3):.2f} GB ({100 - mem.percent}%)")
        
        if pynvml_available and self.resource_logger.initialized:
            for i, handle in enumerate(self.resource_logger.gpu_handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    used_vram = mem_info.used / (1024**3)
                    total_vram = mem_info.total / (1024**3)
                    available_vram = mem_info.free / (1024**3)
                    
                    self.resource_logger.log(
                        f"GPU {i} 状态: "
                        f"使用率 {util.gpu}%, "
                        f"显存(已用/总/可用): {used_vram:.2f} / {total_vram:.2f} / {available_vram:.2f} GB, "
                        f"温度 {temp}°C"
                    )
                except pynvml.NVMLError as e:
                    self.resource_logger.log(f"GPU {i} 信息获取失败: {e}")
        
        process = psutil.Process(os.getpid())
        self.resource_logger.log(f"当前进程内存使用: {process.memory_info().rss / (1024**3):.2f} GB")
        self.resource_logger.log("=================================\n")
    
    def on_train_end(self, logs=None) -> None:
        total_time = time.time() - self.start_time
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        self.resource_logger.log(f"\n训练总耗时: {int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒")
        self.resource_logger.record_memory_summary()


class LayerOutputMonitor(keras.callbacks.Callback):
    """监控模型中间层输出，检测NaN/Inf等异常值"""
    def __init__(self, 
                 model: keras.Model, 
                 layer_names: List[str], 
                 sample_batch: Optional[np.ndarray] = None, 
                 resource_logger: Optional[ResourceLogger] = None):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.resource_logger = resource_logger or ResourceLogger()
        self.valid_layers = [name for name in layer_names if name in self._get_all_layer_names()]
        self.invalid_layers = [name for name in layer_names if name not in self._get_all_layer_names()]
        if self.invalid_layers:
            self.resource_logger.log(f"警告：以下监控层不存在，已自动过滤：{self.invalid_layers}")
        self.feature_extractor = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(name).output for name in self.valid_layers]
        )
        self.sample_batch = sample_batch
        self.nan_detector = NaNDetector(verbose=True)

    def _get_all_layer_names(self) -> List[str]:
        all_names: List[str] = []
        def collect(layer: layers.Layer, parent_name: str = "") -> None:
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

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if self.sample_batch is None or not self.valid_layers:
            return
        
        try:
            # 适配双输入情况
            if isinstance(self.sample_batch, list) and len(self.sample_batch) == 2:
                self.nan_detector.check_nan(self.sample_batch[0], f"Epoch {epoch} 监控批次输入（音频）")
                self.nan_detector.check_nan(self.sample_batch[1], f"Epoch {epoch} 监控批次输入（梅尔）")
            else:
                self.nan_detector.check_nan(self.sample_batch, f"Epoch {epoch} 监控批次输入")
                
            layer_outputs = self.feature_extractor.predict(self.sample_batch, verbose=0)
            
            for name, output in zip(self.valid_layers, layer_outputs):
                self.resource_logger.log(f"\n===== Epoch {epoch} 层 {name} 输出监控 =====")
                self.resource_logger.log(f"形状: {output.shape}")
                self.nan_detector.check_nan(output, f"Epoch {epoch} 层 {name} 输出")
                
                if np.isnan(output).any():
                    self.resource_logger.log("存在NaN值！")
                else:
                    self.resource_logger.log(f"最小值: {np.min(output):.6f}, 最大值: {np.max(output):.6f}")
                self.resource_logger.log(f"含Inf: {np.isinf(output).any()}")
                self.resource_logger.log("=========================================\n")
        except Exception as e:
            self.resource_logger.log(f"层输出监控出错: {str(e)}")
            traceback.print_exc()


class GradientMonitor(keras.callbacks.Callback):
    """修复的梯度监控器，适配分布式训练"""
    def __init__(self, 
                 model: keras.Model, 
                 sample_batch: Tuple[np.ndarray, np.ndarray], 
                 strategy: tf.distribute.Strategy, 
                 resource_logger: Optional[ResourceLogger] = None):
        super().__init__()
        self.model = model
        self.sample_batch = sample_batch  # (x, y)元组，x可能是双输入列表
        self.strategy = strategy  # 传入分布式策略
        self.resource_logger = resource_logger or ResourceLogger()
        self.loss_fn = self.model.loss
        if isinstance(self.loss_fn, str):
            self.loss_fn = keras.losses.get(self.loss_fn)
        self.x_batch, self.y_batch = sample_batch
        # 确保监控批次适配分布式训练
        self.dist_dataset = self._prepare_distributed_dataset()

    def _prepare_distributed_dataset(self) -> tf.data.Dataset:
        """将监控数据转换为分布式数据集，支持双输入"""
        if isinstance(self.x_batch, list) and len(self.x_batch) == 2:
            # 双输入情况
            dataset = tf.data.Dataset.from_tensor_slices(((self.x_batch[0], self.x_batch[1]), self.y_batch))
        else:
            # 单输入情况
            dataset = tf.data.Dataset.from_tensor_slices((self.x_batch, self.y_batch))
        dataset = dataset.batch(len(self.y_batch))  # 单批次
        return self.strategy.experimental_distribute_dataset(dataset)

    def _compute_gradients(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """在副本上下文中计算梯度的函数，支持双输入"""
        x, y = inputs
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
        grads = tape.gradient(loss, self.model.trainable_weights)
        return grads, loss

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """每个epoch结束时监控梯度，减少性能影响"""
        try:
            # 在分布式策略中运行梯度计算
            for x, y in self.dist_dataset:
                distributed_grads, _ = self.strategy.run(
                    self._compute_gradients, 
                    args=((x, y),)
                )
                
                # 聚合所有副本的梯度
                grads = self.strategy.experimental_local_results(distributed_grads)
                all_grads: List[tf.Tensor] = []
                for replica_grads in grads:
                    all_grads.extend(replica_grads)
                
                # 检查梯度异常
                has_nan = False
                has_inf = False
                grad_norms: List[float] = []
                for i, g in enumerate(all_grads):
                    if g is not None:
                        g_np = g.numpy()
                        grad_norms.append(tf.norm(g).numpy())
                        if np.isnan(g_np).any():
                            has_nan = True
                        if np.isinf(g_np).any():
                            has_inf = True
                
                if has_nan:
                    self.resource_logger.log(f"Epoch {epoch} 梯度包含NaN!")
                if has_inf:
                    self.resource_logger.log(f"Epoch {epoch} 梯度包含Inf!")
                if grad_norms:
                    self.resource_logger.log(f"Epoch {epoch} 梯度范数最大值: {np.max(grad_norms):.4f}")
                
        except Exception as e:
            self.resource_logger.log(f"梯度监控出错: {str(e)}")
            traceback.print_exc()


class DeepSoundBaseRNN:
    """RNN基础类，支持动态填充及多GPU训练，新增双分支支持"""
    def __init__(self,
                 batch_size: int = 8,
                 n_epochs: int = 1400,
                 input_size: int = 1800,
                 mel_input_shape: Tuple[int, int] = (100, 40),  # (时间步, 频率bins)
                 set_sample_weights: bool = True,
                 feature_scaling: bool = True,
                 output_size: int = 4):  # 输出类别数
        self.classes_: Optional[List[int]] = None
        self.padding_class: Optional[int] = None  # 填充类别标记
        self.max_seq_len: Optional[int] = None    # 训练时的最大序列长度
        self.input_size = input_size  # 音频输入特征维度
        self.mel_input_shape = mel_input_shape  # 梅尔频谱输入形状 (时间步, 频率bins)
        self.original_lengths: List[int] = []     # 存储训练样本原始长度
        self.output_size = output_size  # 输出维度

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.padding = "same"  # 使用same填充，避免卷积后维度缩小过多
        self.set_sample_weights = set_sample_weights
        self.feature_scaling = feature_scaling
        self.model: Optional[keras.Model] = None
        self.weights_: Optional[List[np.ndarray]] = None
        self.model_save_path = "./model_checkpoints"
        self.nan_detector = NaNDetector(verbose=True)
        self.strategy: Optional[tf.distribute.Strategy] = None
        self.resource_logger = ResourceLogger()
        os.makedirs(self.model_save_path, exist_ok=True)

    def _build_audio_branch(self) -> Sequential:
        """构建音频分支网络（1D CNN + GRU）"""
        # 音频分支CNN配置
        layers_config = [
            (32, 18, 3, activations.relu),
            (32, 9, 1, activations.relu),
            (128, 3, 1, activations.relu)
        ]

        # CNN子网络 - 用于音频特征提取
        cnn = Sequential(name='audio_cnn')
        cnn.add(layers.Rescaling(scale=1.0, name='audio_input_rescaling'))

        for ix_l, layer in enumerate(layers_config):
            # 第一个卷积块
            cnn.add(layers.Conv1D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=None,
                padding=self.padding,
                data_format=self.data_format,
                kernel_initializer=HeUniform(),
                name=f'audio_conv1d_{ix_l*2 + 1}'
            ))
            cnn.add(layers.BatchNormalization(name=f'audio_bn_{ix_l*2 + 1}'))
            cnn.add(layers.Activation(layer[3], name=f'audio_act_{ix_l*2 + 1}'))

            # 第二个卷积块
            cnn.add(layers.Conv1D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=None,
                padding=self.padding,
                data_format=self.data_format,
                kernel_initializer=HeUniform(),
                name=f'audio_conv1d_{ix_l*2 + 2}'
            ))
            cnn.add(layers.BatchNormalization(name=f'audio_bn_{ix_l*2 + 2}'))
            cnn.add(layers.Activation(layer[3], name=f'audio_act_{ix_l*2 + 2}'))

            # 除最后一层外添加Dropout
            if ix_l < (len(layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.3, name=f'audio_dropout_{ix_l + 1}'))

        cnn.add(layers.MaxPooling1D(4, name='audio_max_pooling1d'))
        cnn.add(layers.Flatten(name='audio_flatten'))
        cnn.add(layers.Dropout(rate=0.3, name='audio_cnn_output_dropout'))

        # 音频分支时序建模
        audio_branch = Sequential(name='audio_branch')
        audio_branch.add(layers.TimeDistributed(cnn, name='audio_time_distributed_cnn'))
        audio_branch.add(layers.Bidirectional(
            layers.GRU(128,
                       activation="tanh", 
                       return_sequences=True, 
                       dropout=0.3,
                       recurrent_dropout=0.3,
                       kernel_initializer=HeUniform()),
            name='audio_bidirectional_gru'
        ))
        
        return audio_branch

    def _build_mel_branch(self) -> Sequential:
        """构建梅尔频谱分支网络（2D CNN + GRU）- 修复池化维度问题"""
        # 梅尔分支CNN配置（减少池化次数，避免维度≤0）
        mel_layers_config = [
            (32, (3, 3), (1, 1), activations.relu),
            (32, (3, 3), (1, 1), activations.relu),
            (128, (3, 3), (1, 1), activations.relu)
        ]

        # 2D CNN子网络 - 用于梅尔频谱特征提取
        cnn = Sequential(name='mel_cnn')
        cnn.add(layers.Rescaling(scale=1.0, name='mel_input_rescaling'))

        for ix_l, layer in enumerate(mel_layers_config):
            # 第一个卷积块
            cnn.add(layers.Conv2D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=None,
                padding=self.padding,  # 使用same填充，保持维度
                data_format=self.data_format,
                kernel_initializer=HeUniform(),
                name=f'mel_conv2d_{ix_l*2 + 1}'
            ))
            cnn.add(layers.BatchNormalization(name=f'mel_bn_{ix_l*2 + 1}'))
            cnn.add(layers.Activation(layer[3], name=f'mel_act_{ix_l*2 + 1}'))

            # 第二个卷积块
            cnn.add(layers.Conv2D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=None,
                padding=self.padding,
                data_format=self.data_format,
                kernel_initializer=HeUniform(),
                name=f'mel_conv2d_{ix_l*2 + 2}'
            ))
            cnn.add(layers.BatchNormalization(name=f'mel_bn_{ix_l*2 + 2}'))
            cnn.add(layers.Activation(layer[3], name=f'mel_act_{ix_l*2 + 2}'))

            # 仅最后一层添加池化（避免过度压缩）
            if ix_l == len(mel_layers_config) - 1:
                cnn.add(layers.MaxPooling2D((1, 2), name=f'mel_max_pooling2d_{ix_l + 1}'))  # 仅压缩时间维度
            
            # 除最后一层外添加Dropout
            if ix_l < (len(mel_layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.3, name=f'mel_dropout_{ix_l + 1}'))

        # 展平特征
        cnn.add(layers.GlobalAveragePooling2D(name='mel_global_avg_pool'))
        cnn.add(layers.Dropout(rate=0.3, name='mel_cnn_output_dropout'))

        # 梅尔分支时序建模
        mel_branch = Sequential(name='mel_branch')
        mel_branch.add(layers.TimeDistributed(cnn, name='mel_time_distributed_cnn'))
        mel_branch.add(layers.Bidirectional(
            layers.GRU(128,
                       activation="tanh", 
                       return_sequences=True, 
                       dropout=0.3,
                       recurrent_dropout=0.3,
                       kernel_initializer=HeUniform()),
            name='mel_bidirectional_gru'
        ))
        
        return mel_branch

    def _build_model(self, max_seq_len: int, output_size: int = 4) -> keras.Model:
        """构建双分支模型结构（音频+梅尔频谱）- 修复输入维度"""
        # 打印模型输出维度
        self.resource_logger.log(f"模型构建 - 输出维度output_size={output_size}，最大序列长度max_seq_len={max_seq_len}")
        
        # 1. 定义输入层（修正梅尔输入形状：(序列长度, 时间步, 频率bins, 通道)）
        audio_input = layers.Input(shape=(None, self.input_size, 1), name='audio_input')
        mel_input = layers.Input(shape=(None, self.mel_input_shape[0], self.mel_input_shape[1], 1), name='mel_input')
        
        # 2. 构建分支网络
        audio_branch = self._build_audio_branch()
        mel_branch = self._build_mel_branch()
        
        # 3. 分支特征提取
        audio_features = audio_branch(audio_input)  # 形状: (None, 序列长度, 128)
        mel_features = mel_branch(mel_input)        # 形状: (None, 序列长度, 128)
        
        # 4. 融合分支特征（GRU输出后拼接）
        merged = layers.Concatenate(axis=-1, name='features_concatenation')([audio_features, mel_features])  # 形状: (None, 序列长度, 256)
        
        # 5. 最终分类头
        ffn = Sequential(name='ffn_subnetwork')
        ffn.add(layers.Dense(256, activation=None, kernel_initializer=HeUniform(), name='ffn_dense_1'))
        ffn.add(layers.BatchNormalization(name='ffn_bn_1'))
        ffn.add(layers.Activation(activations.relu, name='ffn_act_1'))
        ffn.add(layers.Dropout(rate=0.3, name='ffn_dropout_1'))
        
        ffn.add(layers.Dense(128, activation=None, kernel_initializer=HeUniform(), name='ffn_dense_2'))
        ffn.add(layers.BatchNormalization(name='ffn_bn_2'))
        ffn.add(layers.Activation(activations.relu, name='ffn_act_2'))
        ffn.add(layers.Dropout(rate=0.3, name='ffn_dropout_2'))
        
        ffn.add(layers.Dense(output_size, activation=activations.softmax, name='ffn_output'))
        
        # 6. 时间分布式分类
        outputs = layers.TimeDistributed(ffn, name='time_distributed_ffn')(merged)
        
        # 7. 构建模型
        model = Model(inputs=[audio_input, mel_input], outputs=outputs, name='deep_sound_two_branch')

        model.compile(
            optimizer=Adam(
                learning_rate=1e-3,
                clipnorm=1.0,
                clipvalue=0.5
            ),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['accuracy']
        )

        return model

    def fit(self, X: Union[List[np.ndarray], np.ndarray], y: Union[List[np.ndarray], np.ndarray]) -> None:
        """训练模型，支持双输入(X包含音频和梅尔频谱数据)"""
        self.nan_detector = NaNDetector(verbose=True)
        training_start_time = time.time()
        try:
            self.resource_logger.log("="*60)
            self.resource_logger.log("【训练数据信息】")
            # 验证双输入格式
            if not isinstance(X, list) or len(X) != 2:
                raise ValueError("双分支模型需要传入包含[音频数据, 梅尔频谱数据]的列表作为X")
                
            audio_X, mel_X = X
            self.resource_logger.log(f"音频数据类型: {type(audio_X)}, 长度: {len(audio_X) if isinstance(audio_X, (list, np.ndarray)) else 'N/A'}")
            self.resource_logger.log(f"梅尔频谱数据类型: {type(mel_X)}, 长度: {len(mel_X) if isinstance(mel_X, (list, np.ndarray)) else 'N/A'}")
            self.resource_logger.log(f"标签数据类型: {type(y)}, 长度: {len(y) if isinstance(y, (list, np.ndarray)) else 'N/A'}")
            self.resource_logger.log("="*60)
            
            # 检查音频和梅尔数据长度是否匹配
            if len(audio_X) != len(mel_X):
                raise ValueError(f"音频样本数({len(audio_X)})与梅尔频谱样本数({len(mel_X)})不匹配")
            if len(audio_X) != len(y):
                raise ValueError(f"样本数({len(audio_X)})与标签数({len(y)})不匹配")
            
            # 处理音频数据
            self.resource_logger.log("\n===== 处理音频数据 =====")
            audio_array: List[np.ndarray] = []
            self.original_lengths = []  # 重置并记录原始长度
            for i, sample in enumerate(audio_X):
                if isinstance(sample, list):
                    sample_array = np.array(sample, dtype='float32')
                    audio_array.append(sample_array)
                    self.original_lengths.append(len(sample_array))
                    self.resource_logger.log(f"音频样本{i}：已转换为数组，形状={sample_array.shape}")
                elif isinstance(sample, np.ndarray):
                    audio_array.append(sample)
                    self.original_lengths.append(len(sample))
                    self.resource_logger.log(f"音频样本{i}：已是数组，形状={sample.shape}")
                else:
                    raise TypeError(f"音频样本{i}：类型错误={type(sample)}")
            audio_X = audio_array
            self.nan_detector.check_nan(audio_X, "转换为数组后的音频数据")
            
            # 处理梅尔频谱数据（强制统一为3维：序列长度, 时间步, 频率bins）
            self.resource_logger.log("\n===== 处理梅尔频谱数据 =====")
            mel_array: List[np.ndarray] = []
            pred_original_lengths = []
            for i, sample in enumerate(mel_X):
                # 转换为数组
                if isinstance(sample, list):
                    sample_array = np.array(sample, dtype='float32')
                elif isinstance(sample, np.ndarray):
                    sample_array = sample.astype('float32')
                else:
                    raise TypeError(f"梅尔样本{i}：类型错误={type(sample)}")
                
                original_shape = sample_array.shape
                self.resource_logger.log(f"梅尔样本{i}：原始形状={original_shape}，维度={sample_array.ndim}")
                
                # 核心修复1：强制去除所有大小为1的维度（无论位置）
                sample_array = np.squeeze(sample_array)
                self.resource_logger.log(f"梅尔样本{i}：去除所有单维度后形状={sample_array.shape}，维度={sample_array.ndim}")
                
                # 核心修复2：确保最终为3维
                if sample_array.ndim == 2:
                    # 2维：(时间步, 频率bins) → 扩展为 (1, 时间步, 频率bins)
                    sample_array = np.expand_dims(sample_array, axis=0)
                    self.resource_logger.log(f"梅尔样本{i}：2维→3维（添加序列长度维度），形状={sample_array.shape}")
                elif sample_array.ndim == 3:
                    # 3维：直接使用
                    self.resource_logger.log(f"梅尔样本{i}：3维（格式正确），形状={sample_array.shape}")
                elif sample_array.ndim > 3:
                    # 高维：合并前两维为序列长度
                    new_seq_len = sample_array.shape[0] * sample_array.shape[1]
                    new_shape = (new_seq_len, sample_array.shape[2], sample_array.shape[3])
                    sample_array = sample_array.reshape(new_shape)
                    self.resource_logger.log(f"梅尔样本{i}：高维→3维（合并前两维），形状={sample_array.shape}")
                else:  # 1维（异常）
                    # 强制构造3维：(1, 1, 特征数)
                    sample_array = np.expand_dims(np.expand_dims(sample_array, axis=0), axis=0)
                    self.resource_logger.log(f"梅尔样本{i}：1维→强制3维，形状={sample_array.shape}")
                
                pred_original_lengths.append(sample_array.shape[0])  # 记录序列长度（3维的第0维）
                mel_array.append(sample_array)
            
            # 核心修复3：处理后全局检查，确保所有样本都是3维
            for i, mel_sample in enumerate(mel_array):
                if mel_sample.ndim != 3:
                    raise ValueError(f"梅尔样本{i}处理后仍为{mel_sample.ndim}维，形状={mel_sample.shape}（最终检查失败）")
            self.resource_logger.log("所有梅尔样本处理后均为3维，格式正确")
            
            mel_X = mel_array
            self.nan_detector.check_nan(mel_X, "转换为数组后的梅尔数据")
            
            # 处理标签
            self.resource_logger.log("\n===== 处理标签数据 =====")
            if isinstance(y, list) and len(y) == 1 and isinstance(y[0], (list, np.ndarray)):
                y = y[0]
                self.nan_detector.log_process("提取嵌套y样本")
            self.nan_detector.check_nan(y, "原始y数据")
            
            # 统一样本维度
            # 音频数据：确保是2D (序列长度, 特征维度)
            audio_X = [
                np.expand_dims(sample, axis=-1) if (isinstance(sample, np.ndarray) and sample.ndim == 1) 
                else np.squeeze(sample, axis=-1) if (isinstance(sample, np.ndarray) and sample.ndim == 3 and sample.shape[-1] == 1)
                else sample 
                for sample in audio_X
            ]
            
#             # 梅尔数据：确保是3D (序列长度, 时间步, 频率bins)
#             mel_X = [
#                 np.expand_dims(sample, axis=-1) if (isinstance(sample, np.ndarray) and sample.ndim == 3)
#                 else np.squeeze(sample, axis=-1) if (isinstance(sample, np.ndarray) and sample.ndim == 5 and sample.shape[-1] == 1)
#                 else sample
#                 for sample in mel_X
#             ]
            
            # 计算训练集最大序列长度
            if isinstance(audio_X, (list, np.ndarray)):
                self.max_seq_len = max(sample.shape[0] for sample in audio_X) if audio_X else 0
                self.resource_logger.log(f"基于音频样本计算的最大序列长度: {self.max_seq_len}")
            else:
                raise ValueError("音频数据必须是列表或NumPy数组")

            # 同步填充/截断音频数据
            target_len = self.max_seq_len
            audio_padded: List[np.ndarray] = []
            for i, sample in enumerate(audio_X):
                if not isinstance(sample, np.ndarray) or sample.ndim != 2:
                    raise ValueError(f"音频样本必须是2维数组，实际形状: {sample.shape if isinstance(sample, np.ndarray) else type(sample)}")
                
                seq_len, feat_dim = sample.shape  # 音频是2维：(序列长度, 特征维度)
                
                # 统一音频特征维度
                if feat_dim != self.input_size:
                    self.resource_logger.log(f"音频样本{i}特征维度不匹配: 实际{feat_dim}，预期{self.input_size}，自动调整")
                    if feat_dim > self.input_size:
                        sample = sample[:, :self.input_size]  # 截断特征维度
                    else:
                        sample = np.pad(sample, ((0,0), (0, self.input_size - feat_dim)), mode='constant')
                
                # 填充或截断序列长度
                if seq_len < target_len:
                    pad_length = target_len - seq_len
                    padded = np.pad(sample, pad_width=((0, pad_length), (0, 0)),
                                   mode='constant', constant_values=-1.0)
                    self.resource_logger.log(f"音频样本{i}：序列较短（{seq_len} < {target_len}），填充{pad_length}个窗口")
                else:
                    padded = sample[:target_len, :]
                    self.resource_logger.log(f"音频样本{i}：序列较长（{seq_len} > {target_len}），截断至{target_len}个窗口")
                
                audio_padded.append(padded)
            
            # 同步填充/截断梅尔数据（修复维度判断和时间维度强制≥16）
            mel_padded: List[np.ndarray] = []
            # 验证梅尔输入形状格式
            if not isinstance(self.mel_input_shape, tuple) or len(self.mel_input_shape) != 2:
                raise ValueError(f"mel_input_shape 必须是二元组 (时间步, 频率bins)，实际为 {self.mel_input_shape}")
            expected_time, expected_freq = self.mel_input_shape  # (时间步, 频率bins)
            
            for i, sample in enumerate(mel_X):
                # 核心修复4：填充/截断前再次确认3维（避免中间处理意外改变维度）
                if not isinstance(sample, np.ndarray) or sample.ndim != 3:
                    raise ValueError(f"梅尔样本{i}在填充前维度错误，实际形状: {sample.shape if isinstance(sample, np.ndarray) else type(sample)}")
                
                seq_len, time_dim, freq_dim = sample.shape  # 梅尔是3维：(序列长度, 时间步, 频率bins)
                self.resource_logger.log(f"梅尔样本{i}原始维度: (序列长度={seq_len}, 时间步={time_dim}, 频率bins={freq_dim})")
                
                # 强制时间维度≥16，避免卷积池化后维度≤0
                if time_dim < 16:
                    pad_time = 16 - time_dim
                    sample = np.pad(sample, ((0, 0), (0, pad_time), (0, 0)), mode='constant')
                    time_dim = 16
                    self.resource_logger.log(f"梅尔样本{i}：时间步过小（{time_dim - pad_time} < 16），填充至16")
                
                # # 调整时间步维度至预期值
                # if time_dim != expected_time:
                #     self.resource_logger.log(f"梅尔样本{i}时间步维度不匹配: 实际{time_dim}，预期{expected_time}，自动调整")
                #     if time_dim > expected_time:
                #         sample = sample[:, :expected_time, :]  # 截断
                #     else:
                #         sample = np.pad(sample, ((0,0), (0, expected_time - time_dim), (0,0)), mode='constant')
                #     time_dim = expected_time  # 更新为预期值
                
                # 调整频率bins维度至预期值
                if freq_dim != expected_freq:
                    self.resource_logger.log(f"梅尔样本{i}频率维度不匹配: 实际{freq_dim}，预期{expected_freq}，自动调整")
                    if freq_dim > expected_freq:
                        sample = sample[:, :, :expected_freq]  # 截断
                    else:
                        sample = np.pad(sample, ((0,0), (0,0), (0, expected_freq - freq_dim)), mode='constant')
                    freq_dim = expected_freq  # 更新为预期值
                
                # 填充或截断序列长度（与音频保持一致）
                if seq_len < target_len:
                    pad_length = target_len - seq_len
                    padded = np.pad(sample, pad_width=((0, pad_length), (0, 0), (0, 0)),
                                   mode='constant', constant_values=-1.0)
                    self.resource_logger.log(f"梅尔样本{i}：序列较短（{seq_len} < {target_len}），填充{pad_length}个窗口")
                else:
                    padded = sample[:target_len, :, :]
                    self.resource_logger.log(f"梅尔样本{i}：序列较长（{seq_len} > {target_len}），截断至{target_len}个窗口")
                
                mel_padded.append(padded)
            
            # 转换为数组并添加通道维度
            try:
                audio_X = np.array(audio_padded, dtype='float32')
                mel_X = np.array(mel_padded, dtype='float32')
            except ValueError as e:
                self.resource_logger.log(f"转换为数组失败: {e}")
                raise
            
            # 添加通道维度（适应CNN输入）
            if audio_X.ndim == 3:
                audio_X = np.expand_dims(audio_X, axis=-1)  # 形状: (样本数, 序列长度, 特征维度, 1)
            if mel_X.ndim == 3:
                mel_X = np.expand_dims(mel_X, axis=-1)      # 形状: (样本数, 序列长度, 时间步, 频率bins, 1)
                
            self.resource_logger.log(f"音频数据最终形状: {audio_X.shape}")
            self.resource_logger.log(f"梅尔数据最终形状: {mel_X.shape}")
            self.nan_detector.check_nan(audio_X, "处理后的音频数据")
            self.nan_detector.check_nan(mel_X, "处理后的梅尔数据")

            # 处理标签和填充类别
            self.classes_ = list(set(np.concatenate(y))) if y and isinstance(y[0], (list, np.ndarray)) else []
            self.padding_class = max(self.classes_) + 1 if self.classes_ else 0
            self.resource_logger.log(f"训练标签中的类别: {self.classes_}")
            self.resource_logger.log(f"填充类别编号: {self.padding_class}")
            
            # 填充/截断标签
            y_padded = []
            for label_seq in y:
                seq_len = len(label_seq)
                if seq_len < target_len:
                    pad_length = target_len - seq_len
                    padded = np.pad(label_seq, pad_width=(0, pad_length),
                                   mode='constant', constant_values=self.padding_class)
                else:
                    padded = label_seq[:target_len]
                y_padded.append(padded)
            y = np.array(y_padded, dtype='int32')
            self.resource_logger.log(f"标签最终形状: {y.shape}")
            self.nan_detector.check_nan(y, "处理后的标签数据")

            # 特征标准化
            if self.feature_scaling:
                # 音频数据标准化
                audio_non_pad_mask = audio_X != -1.0
                if np.any(audio_non_pad_mask):
                    audio_mean = np.mean(audio_X[audio_non_pad_mask])
                    audio_std = np.std(audio_X[audio_non_pad_mask])
                    audio_X[audio_non_pad_mask] = (audio_X[audio_non_pad_mask] - audio_mean) / (audio_std + 1e-8)
                    self.resource_logger.log(f"音频标准化后统计: min={np.min(audio_X):.4f}, max={np.max(audio_X):.4f}")
                
                # 梅尔数据标准化
                mel_non_pad_mask = mel_X != -1.0
                if np.any(mel_non_pad_mask):
                    mel_mean = np.mean(mel_X[mel_non_pad_mask])
                    mel_std = np.std(mel_X[mel_non_pad_mask])
                    mel_X[mel_non_pad_mask] = (mel_X[mel_non_pad_mask] - mel_mean) / (mel_std + 1e-8)
                    self.resource_logger.log(f"梅尔标准化后统计: min={np.min(mel_X):.4f}, max={np.max(mel_X):.4f}")

            # 确定输出维度
            output_size = self.output_size
            
            # 分布式训练设置
            self.strategy = tf.distribute.MirroredStrategy()
            self.resource_logger.log(f"已检测到 {self.strategy.num_replicas_in_sync} 个GPU，将用于分布式训练")
            
            with self.strategy.scope():
                self.model = self._build_model(max_seq_len=self.max_seq_len, output_size=output_size)
            
            self.weights_ = copy.deepcopy(self.model.get_weights())
            self.resource_logger.log("\n模型初始化完成（多GPU支持），结构如下：")
            self.model.summary(print_fn=self.resource_logger.log)

            # 准备监控批次
            monitor_batch_size = min(self.batch_size, audio_X.shape[0])
            monitor_audio = audio_X[:monitor_batch_size]
            monitor_mel = mel_X[:monitor_batch_size]
            monitor_y = y[:monitor_batch_size]
            monitor_x = [monitor_audio, monitor_mel]
            monitor_batch = (monitor_x, monitor_y)
            self.resource_logger.log(f"监控批次 - 音频: {monitor_audio.shape}, 梅尔: {monitor_mel.shape}, 标签: {monitor_y.shape}")

            # 验证集设置
            use_validation = audio_X.shape[0] >= 5
            validation_split = 0.2 if use_validation else 0.0
            monitor_loss = 'val_loss' if use_validation else 'loss'
            monitor_acc = 'val_accuracy' if use_validation else 'accuracy'
            self.resource_logger.log(f"使用验证集: {use_validation}, 验证比例: {validation_split}")

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
                    layer_names=['audio_time_distributed_cnn', 'mel_time_distributed_cnn', 
                                'audio_bidirectional_gru', 'mel_bidirectional_gru',
                                'features_concatenation', 'time_distributed_ffn'],
                    sample_batch=monitor_x,
                    resource_logger=self.resource_logger
                ),
                ReduceLROnPlateau(
                    monitor=monitor_loss, factor=0.5, patience=15, min_lr=1e-8, verbose=1
                ),
                GradientMonitor(
                    model=self.model,
                    sample_batch=monitor_batch,
                    strategy=self.strategy,
                    resource_logger=self.resource_logger
                ),
                GPUUsageMonitor(interval=10, resource_logger=self.resource_logger)
            ]

            # 样本权重
            sample_weights: Optional[np.ndarray] = None
            if self.set_sample_weights and y.size > 0:
                sample_weights = self._get_samples_weights(y)
                sample_weights = np.clip(sample_weights, 0.0, 10.0)
                self.resource_logger.log(f"样本权重范围: [{np.min(sample_weights):.4f}, {np.max(sample_weights):.4f}]")

            # 开始训练
            self.resource_logger.log(f"\n【开始训练】样本数: {audio_X.shape[0]}, 批次大小: {self.batch_size}, GPU数量: {self.strategy.num_replicas_in_sync}")
            history = self.model.fit(
                x=[audio_X, mel_X],  # 双输入
                y=y,
                epochs=self.n_epochs,
                verbose=1,
                batch_size=self.batch_size,
                validation_split=validation_split,
                shuffle=True,
                sample_weight=sample_weights,
                callbacks=model_callbacks
            )

            # 训练总结
            training_time = time.time() - training_start_time
            training_summary: Dict[str, str] = {
                "总训练轮次": str(self.n_epochs),
                "实际训练轮次": str(len(history.history['loss'])),
                "最终训练损失": f"{history.history['loss'][-1]:.4f}",
                "最终训练准确率": f"{history.history['accuracy'][-1]:.4f}",
                "总训练时间": f"{training_time:.2f}秒"
            }
            
            if use_validation:
                training_summary["最终验证损失"] = f"{history.history['val_loss'][-1]:.4f}"
                training_summary["最终验证准确率"] = f"{history.history['val_accuracy'][-1]:.4f}"
                
                best_val_loss_idx = np.argmin(history.history['val_loss'])
                best_val_acc_idx = np.argmax(history.history['val_accuracy'])
                training_summary["最佳验证损失"] = f"{history.history['val_loss'][best_val_loss_idx]:.4f} (在第{best_val_loss_idx+1}轮)"
                training_summary["最佳验证准确率"] = f"{history.history['val_accuracy'][best_val_acc_idx]:.4f} (在第{best_val_acc_idx+1}轮)"

            self.resource_logger.record_training_summary(training_summary)
            
        except Exception as e:
            self.resource_logger.log(f"训练过程出错: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            self.resource_logger.save_log()

    def predict(self, X: Union[List[np.ndarray], np.ndarray], aggregate: bool = False) -> Union[List[np.ndarray], np.ndarray]:
        """模型预测，支持双输入(X包含音频和梅尔频谱数据)"""
        try:
            if self.max_seq_len is None:
                raise RuntimeError("请先调用fit方法训练模型")
            
            # 验证双输入格式
            if not isinstance(X, list) or len(X) != 2:
                raise ValueError("双分支模型预测需要传入包含[音频数据, 梅尔频谱数据]的列表作为X")
                
            audio_X, mel_X = X
            pred_detector = NaNDetector(verbose=True)
            self.resource_logger.log("\n===== 开始预测 =====")
            
            # 处理音频数据
            audio_array: List[np.ndarray] = []
            pred_original_lengths = []
            for i, sample in enumerate(audio_X):
                if isinstance(sample, list):
                    sample_array = np.array(sample, dtype='float32')
                    audio_array.append(sample_array)
                    pred_original_lengths.append(sample_array.shape[0])
                    self.resource_logger.log(f"预测音频样本{i}：形状={sample_array.shape}")
                elif isinstance(sample, np.ndarray):
                    audio_array.append(sample)
                    pred_original_lengths.append(sample.shape[0])
                    self.resource_logger.log(f"预测音频样本{i}：形状={sample.shape}")
                else:
                    raise TypeError(f"预测音频样本{i}：类型错误={type(sample)}")
            audio_X = audio_array
            
            # 处理梅尔数据（强制3维）
            mel_array: List[np.ndarray] = []
            for i, sample in enumerate(mel_X):
                if isinstance(sample, list):
                    sample_array = np.array(sample, dtype='float32')
                elif isinstance(sample, np.ndarray):
                    sample_array = sample.astype('float32')
                else:
                    raise TypeError(f"预测梅尔样本{i}：类型错误={type(sample)}")
                
                # 处理4维数据：优先去除所有大小为1的维度
                original_shape = sample_array.shape
                sample_array = np.squeeze(sample_array)
                self.resource_logger.log(f"预测梅尔样本{i}：原始形状={original_shape}，去除单维度后形状={sample_array.shape}")
                
                # 确保3维
                if sample_array.ndim == 2:
                    sample_array = np.expand_dims(sample_array, axis=0)
                    self.resource_logger.log(f"预测梅尔样本{i}：2维→3维，形状={sample_array.shape}")
                elif sample_array.ndim > 3:
                    new_seq_len = sample_array.shape[0] * sample_array.shape[1]
                    new_shape = (new_seq_len, sample_array.shape[2], sample_array.shape[3])
                    sample_array = sample_array.reshape(new_shape)
                    self.resource_logger.log(f"预测梅尔样本{i}：高维→3维，形状={sample_array.shape}")
                elif sample_array.ndim != 3:
                    raise ValueError(f"预测梅尔样本{i}：维度错误，需2、3或4维，实际{sample_array.ndim}维")
                
                mel_array.append(sample_array)
                self.resource_logger.log(f"预测梅尔样本{i}：形状={sample_array.shape}")
            mel_X = mel_array
            
            # 检查长度匹配
            if len(audio_X) != len(mel_X):
                raise ValueError(f"预测音频样本数({len(audio_X)})与梅尔样本数({len(mel_X)})不匹配")

            # 统一样本维度
            audio_X = [
                np.expand_dims(sample, axis=-1) if sample.ndim == 1 
                else np.squeeze(sample, axis=-1) if (sample.ndim == 3 and sample.shape[-1] == 1)
                else sample 
                for sample in audio_X
            ]
            
            mel_X = [
                np.expand_dims(sample, axis=-1) if sample.ndim == 3 
                else np.squeeze(sample, axis=-1) if (sample.ndim == 5 and sample.shape[-1] == 1)
                else sample 
                for sample in mel_X
            ]

            # 填充/截断到训练时的最大序列长度
            target_len = self.max_seq_len
            audio_padded: List[np.ndarray] = []
            for i, sample in enumerate(audio_X):
                if sample.ndim != 2:
                    raise ValueError(f"音频样本必须是2维数组，实际形状: {sample.shape}")
                
                seq_len, feat_dim = sample.shape
                # 统一特征维度
                if feat_dim != self.input_size:
                    if feat_dim > self.input_size:
                        sample = sample[:, :self.input_size]
                    else:
                        sample = np.pad(sample, ((0,0), (0, self.input_size-feat_dim)), mode='constant')
                
                # 填充或截断
                if seq_len < target_len:
                    pad_length = target_len - seq_len
                    padded = np.pad(sample, pad_width=((0, pad_length), (0, 0)),
                                   mode='constant', constant_values=-1.0)
                else:
                    padded = sample[:target_len, :]
                
                audio_padded.append(padded)
            
            # 梅尔数据填充/截断
            mel_padded: List[np.ndarray] = []
            expected_time, expected_freq = self.mel_input_shape  # (时间步, 频率bins)
            for i, sample in enumerate(mel_X):
                if sample.ndim != 3:
                    raise ValueError(f"梅尔样本必须是3维数组，实际形状: {sample.shape}")
                
                seq_len, time_dim, freq_dim = sample.shape
                # 强制时间维度≥16
                if time_dim < 16:
                    pad_time = 16 - time_dim
                    sample = np.pad(sample, ((0, 0), (0, pad_time), (0, 0)), mode='constant')
                    time_dim = 16
                
                # 统一特征维度
                if (time_dim, freq_dim) != (expected_time, expected_freq):
                    if time_dim > expected_time:
                        sample = sample[:, :expected_time, :]
                    else:
                        sample = np.pad(sample, ((0, 0), (0, expected_time - time_dim), (0, 0)), mode='constant')
                    if freq_dim > expected_freq:
                        sample = sample[:, :, :expected_freq]
                    else:
                        sample = np.pad(sample, ((0, 0), (0, 0), (0, expected_freq - freq_dim)), mode='constant')
                
                # 填充或截断序列长度
                if seq_len < target_len:
                    pad_length = target_len - seq_len
                    padded = np.pad(sample, pad_width=((0, pad_length), (0, 0), (0, 0)),
                                   mode='constant', constant_values=-1.0)
                else:
                    padded = sample[:target_len, :, :]
                
                mel_padded.append(padded)
            
            # 转换为数组并添加通道维度
            audio_X = np.array(audio_padded, dtype='float32')
            mel_X = np.array(mel_padded, dtype='float32')
            
            if audio_X.ndim == 3:
                audio_X = np.expand_dims(audio_X, axis=-1)
            if mel_X.ndim == 3:
                mel_X = np.expand_dims(mel_X, axis=-1)
            
            # 标准化
            if self.feature_scaling:
                # 音频标准化
                audio_non_pad_mask = audio_X != -1.0
                if np.any(audio_non_pad_mask):
                    audio_mean = np.mean(audio_X[audio_non_pad_mask])
                    audio_std = np.std(audio_X[audio_non_pad_mask])
                    audio_X[audio_non_pad_mask] = (audio_X[audio_non_pad_mask] - audio_mean) / (audio_std + 1e-8)
                
                # 梅尔标准化
                mel_non_pad_mask = mel_X != -1.0
                if np.any(mel_non_pad_mask):
                    mel_mean = np.mean(mel_X[mel_non_pad_mask])
                    mel_std = np.std(mel_X[mel_non_pad_mask])
                    mel_X[mel_non_pad_mask] = (mel_X[mel_non_pad_mask] - mel_mean) / (mel_std + 1e-8)

            # 模型预测
            self.resource_logger.log(f"预测输入 - 音频形状: {audio_X.shape}, 梅尔形状: {mel_X.shape}")
            assert self.model is not None, "模型未初始化，请先训练模型"
            y_pred_proba = self.model.predict([audio_X, mel_X], verbose=0)  # 双输入预测
            
            # 处理标签4的概率
            if y_pred_proba.shape[-1] > 4:
                y_pred_proba[..., 4] = 0.0
                row_sums = y_pred_proba.sum(axis=-1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                y_pred_proba = y_pred_proba / row_sums
            
            y_pred = y_pred_proba.argmax(axis=-1)

            # 根据原始长度裁剪
            trimmed_preds = []
            for i in range(len(y_pred)):
                real_len = pred_original_lengths[i]
                clip_len = min(real_len, y_pred.shape[1])
                trimmed = y_pred[i, :clip_len]
                trimmed_preds.append(trimmed)
            
            # 聚合序列结果
            if aggregate:
                self.resource_logger.log("开始聚合预测结果...")
                aggregated = []
                for i, seq in enumerate(trimmed_preds):
                    if len(seq) == 0:
                        aggregated.append(0)
                        continue
                    
                    counts = np.bincount(seq)
                    most_common = np.argmax(counts)
                    aggregated.append(most_common)
                
                result = np.array(aggregated, dtype=int)
                self.resource_logger.log(f"预测完成，聚合后结果形状: {result.shape}")
                return result
            else:
                flat_predictions = np.concatenate(trimmed_preds) if trimmed_preds else np.array([])
                self.resource_logger.log(f"预测完成，窗口级结果总长度: {len(flat_predictions)}")
                return flat_predictions

        except Exception as e:
            self.resource_logger.log(f"预测过程出错: {str(e)}")
            traceback.print_exc()
            raise

    def predict_proba(self, X: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        """预测概率，支持双输入 - 补全缺失的处理逻辑"""
        try:
            if self.max_seq_len is None:
                raise RuntimeError("请先调用fit方法训练模型")
            
            # 验证双输入格式
            if not isinstance(X, list) or len(X) != 2:
                raise ValueError("双分支模型预测概率需要传入包含[音频数据, 梅尔频谱数据]的列表作为X")
                
            audio_X, mel_X = X
            pred_detector = NaNDetector(verbose=True)
            self.resource_logger.log("\n===== 开始预测概率 =====")
            
            # 处理音频数据
            audio_array: List[np.ndarray] = []
            pred_original_lengths = []
            for i, sample in enumerate(audio_X):
                if isinstance(sample, list):
                    sample_array = np.array(sample, dtype='float32')
                    audio_array.append(sample_array)
                    pred_original_lengths.append(sample_array.shape[0])
                    self.resource_logger.log(f"预测音频样本{i}：形状={sample_array.shape}")
                elif isinstance(sample, np.ndarray):
                    audio_array.append(sample)
                    pred_original_lengths.append(sample.shape[0])
                    self.resource_logger.log(f"预测音频样本{i}：形状={sample.shape}")
                else:
                    raise TypeError(f"预测音频样本{i}：类型错误={type(sample)}")
            audio_X = audio_array
            
            # 处理梅尔数据（强制3维）
            mel_array: List[np.ndarray] = []
            for i, sample in enumerate(mel_X):
                if isinstance(sample, list):
                    sample_array = np.array(sample, dtype='float32')
                elif isinstance(sample, np.ndarray):
                    sample_array = sample.astype('float32')
                else:
                    raise TypeError(f"预测梅尔样本{i}：类型错误={type(sample)}")
                
                # 处理4维数据：优先去除所有大小为1的维度
                original_shape = sample_array.shape
                sample_array = np.squeeze(sample_array)
                self.resource_logger.log(f"预测梅尔样本{i}：原始形状={original_shape}，去除单维度后形状={sample_array.shape}")
                
                # 确保3维
                if sample_array.ndim == 2:
                    sample_array = np.expand_dims(sample_array, axis=0)
                    self.resource_logger.log(f"预测梅尔样本{i}：2维→3维，形状={sample_array.shape}")
                elif sample_array.ndim > 3:
                    new_seq_len = sample_array.shape[0] * sample_array.shape[1]
                    new_shape = (new_seq_len, sample_array.shape[2], sample_array.shape[3])
                    sample_array = sample_array.reshape(new_shape)
                    self.resource_logger.log(f"预测梅尔样本{i}：高维→3维，形状={sample_array.shape}")
                elif sample_array.ndim != 3:
                    raise ValueError(f"预测梅尔样本{i}：维度错误，需2、3或4维，实际{sample_array.ndim}维")
                
                mel_array.append(sample_array)
                self.resource_logger.log(f"预测梅尔样本{i}：形状={sample_array.shape}")
            mel_X = mel_array
            
            # 检查长度匹配
            if len(audio_X) != len(mel_X):
                raise ValueError(f"预测音频样本数({len(audio_X)})与梅尔样本数({len(mel_X)})不匹配")

            # 统一样本维度
            audio_X = [
                np.expand_dims(sample, axis=-1) if sample.ndim == 1 
                else np.squeeze(sample, axis=-1) if (sample.ndim == 3 and sample.shape[-1] == 1)
                else sample 
                for sample in audio_X
            ]
            
            mel_X = [
                np.expand_dims(sample, axis=-1) if sample.ndim == 3 
                else np.squeeze(sample, axis=-1) if (sample.ndim == 5 and sample.shape[-1] == 1)
                else sample 
                for sample in mel_X
            ]

            # 填充/截断到训练时的最大序列长度
            target_len = self.max_seq_len
            audio_padded: List[np.ndarray] = []
            for i, sample in enumerate(audio_X):
                if sample.ndim != 2:
                    raise ValueError(f"音频样本必须是2维数组，实际形状: {sample.shape}")
                
                seq_len, feat_dim = sample.shape
                # 统一特征维度
                if feat_dim != self.input_size:
                    if feat_dim > self.input_size:
                        sample = sample[:, :self.input_size]
                    else:
                        sample = np.pad(sample, ((0,0), (0, self.input_size-feat_dim)), mode='constant')
                
                # 填充或截断
                if seq_len < target_len:
                    pad_length = target_len - seq_len
                    padded = np.pad(sample, pad_width=((0, pad_length), (0, 0)),
                                   mode='constant', constant_values=-1.0)
                else:
                    padded = sample[:target_len, :]
                
                audio_padded.append(padded)
            
            # 梅尔数据填充/截断
            mel_padded: List[np.ndarray] = []
            expected_time, expected_freq = self.mel_input_shape  # (时间步, 频率bins)
            for i, sample in enumerate(mel_X):
                if sample.ndim != 3:
                    raise ValueError(f"梅尔样本必须是3维数组，实际形状: {sample.shape}")
                
                seq_len, time_dim, freq_dim = sample.shape
                # 强制时间维度≥16
                if time_dim < 16:
                    pad_time = 16 - time_dim
                    sample = np.pad(sample, ((0, 0), (0, pad_time), (0, 0)), mode='constant')
                    time_dim = 16
                
                # 统一特征维度
                if (time_dim, freq_dim) != (expected_time, expected_freq):
                    if time_dim > expected_time:
                        sample = sample[:, :expected_time, :]
                    else:
                        sample = np.pad(sample, ((0, 0), (0, expected_time - time_dim), (0, 0)), mode='constant')
                    if freq_dim > expected_freq:
                        sample = sample[:, :, :expected_freq]
                    else:
                        sample = np.pad(sample, ((0, 0), (0, 0), (0, expected_freq - freq_dim)), mode='constant')
                
                # 填充或截断序列长度
                if seq_len < target_len:
                    pad_length = target_len - seq_len
                    padded = np.pad(sample, pad_width=((0, pad_length), (0, 0), (0, 0)),
                                   mode='constant', constant_values=-1.0)
                else:
                    padded = sample[:target_len, :, :]
                
                mel_padded.append(padded)
            
            # 转换为数组并添加通道维度
            audio_X = np.array(audio_padded, dtype='float32')
            mel_X = np.array(mel_padded, dtype='float32')
            
            if audio_X.ndim == 3:
                audio_X = np.expand_dims(audio_X, axis=-1)
            if mel_X.ndim == 3:
                mel_X = np.expand_dims(mel_X, axis=-1)
            
            # 标准化
            if self.feature_scaling:
                # 音频标准化
                audio_non_pad_mask = audio_X != -1.0
                if np.any(audio_non_pad_mask):
                    audio_mean = np.mean(audio_X[audio_non_pad_mask])
                    audio_std = np.std(audio_X[audio_non_pad_mask])
                    audio_X[audio_non_pad_mask] = (audio_X[audio_non_pad_mask] - audio_mean) / (audio_std + 1e-8)
                
                # 梅尔标准化                
                mel_non_pad_mask = mel_X != -1.0
                if np.any(mel_non_pad_mask):
                    mel_mean = np.mean(mel_X[mel_non_pad_mask])
                    mel_std = np.std(mel_X[mel_non_pad_mask])
                    mel_X[mel_non_pad_mask] = (mel_X[mel_non_pad_mask] - mel_mean) / (mel_std + 1e-8)

            # 模型预测概率
            self.resource_logger.log(f"预测概率输入 - 音频形状: {audio_X.shape}, 梅尔形状: {mel_X.shape}")
            assert self.model is not None, "模型未初始化，请先训练模型"
            y_pred_proba = self.model.predict([audio_X, mel_X], verbose=0)  # 双输入预测
            
            # 处理标签4的概率（与predict方法逻辑一致）
            if y_pred_proba.shape[-1] > 4:
                y_pred_proba[..., 4] = 0.0
                row_sums = y_pred_proba.sum(axis=-1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                y_pred_proba = y_pred_proba / row_sums
            
            # 根据原始长度裁剪概率序列
            trimmed_probs = []
            for i in range(len(y_pred_proba)):
                real_len = pred_original_lengths[i]
                clip_len = min(real_len, y_pred_proba.shape[1])
                trimmed = y_pred_proba[i, :clip_len, :]  # 保留所有类别概率
                trimmed_probs.append(trimmed)
            
            self.resource_logger.log(f"预测概率完成，裁剪后结果数量: {len(trimmed_probs)}")
            return trimmed_probs if len(trimmed_probs) > 1 else trimmed_probs[0]

        except Exception as e:
            self.resource_logger.log(f"预测概率过程出错: {str(e)}")
            traceback.print_exc()
            raise

    def _get_samples_weights(self, y: np.ndarray) -> np.ndarray:
        """计算样本权重，填充类别权重为0"""
        unique_classes, counts = np.unique(np.ravel(y), return_counts=True)
        counts = np.maximum(counts, 1)  # 避免除以0
        
        # 计算类别权重（平衡类别）
        class_weight = np.log((counts.max() / counts) + 1.0)
        
        # 填充类别权重设为0
        if self.padding_class in unique_classes:
            pad_idx = np.where(unique_classes == self.padding_class)[0][0]
            class_weight[pad_idx] = 0.0
        
        # 日志输出
        self.resource_logger.log("\n===== 类别权重 =====")
        for cls, cnt, weight in zip(unique_classes, counts, class_weight):
            cls_type = "填充类别" if cls == self.padding_class else "普通类别"
            self.resource_logger.log(f"类别 {cls} ({cls_type}): 样本数={cnt}, 权重={weight:.4f}")
        self.resource_logger.log("====================\n")

        # 生成样本权重矩阵
        sample_weight = np.zeros_like(y, dtype=float)
        for class_num, weight in zip(unique_classes, class_weight):
            sample_weight[y == class_num] = weight

        return sample_weight

    def clear_params(self) -> None:
        """重置模型权重"""
        if self.weights_ is not None and self.model is not None:
            self.model.set_weights(copy.deepcopy(self.weights_))
            self.resource_logger.log("模型权重已重置为初始状态")
        else:
            self.resource_logger.log("警告：未初始化模型权重，无法重置")


class DeepSound(DeepSoundBaseRNN):
    """DeepSound双分支模型，继承自RNN基础类"""
    def __init__(self,
                 batch_size: int = 5,
                 input_size: int = 4000,
                 mel_input_shape: Tuple[int, int] = (100, 40),  # (时间步, 频率bins)
                 output_size: int = 3,
                 n_epochs: int = 1400,
                 training_reshape: bool = False,
                 set_sample_weights: bool = True,
                 feature_scaling: bool = True):
        super().__init__(
            batch_size=batch_size,
            n_epochs=n_epochs,
            input_size=input_size,
            mel_input_shape=mel_input_shape,  # 传递梅尔输入形状
            set_sample_weights=set_sample_weights,
            feature_scaling=feature_scaling,
            output_size=output_size
        )
        self.training_reshape = training_reshape


# 补充：如果需要单独运行测试，可添加以下代码（可选）
if __name__ == "__main__":
    # 测试模型初始化
    model = DeepSound(
        batch_size=2,
        input_size=4000,
        mel_input_shape=(100, 40),
        output_size=3,
        n_epochs=10
    )
    print("模型初始化成功")