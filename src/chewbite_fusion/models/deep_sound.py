import os
import copy
import time
import numpy as np
import tensorflow as tf
import psutil
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Union, Dict
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad  # 保持代码二使用的优化器
from tensorflow.keras import activations
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
# 假设NaNDetector的路径正确
from chewbite_fusion.data.utils import NaNDetector
import traceback

# 初始化模块日志（采用代码二的日志配置）
logger = logging.getLogger('yaer')
logging.basicConfig(level=logging.INFO)


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
        self.used_devices: List[str] = []  # 记录使用的设备
        
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
        # 同时输出到logging（代码二功能）
        logger.info(message)
    
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
        
        # 记录可用计算设备
        self.log("\n===== 可用计算设备 =====")
        physical_devices = tf.config.list_physical_devices()
        for device in physical_devices:
            self.log(f"设备: {device.name}, 类型: {device.device_type}")
            self.used_devices.append(device.name)
        
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
        
        # 记录实际使用的设备
        self.log("\n===== 实际使用的设备 =====")
        for device in self.used_devices:
            self.log(f"设备: {device}")
        
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
        self.batch_counter = 0
        
    def on_train_begin(self, logs=None) -> None:
        self.start_time = time.time()
        self.resource_logger.log("\n===== 系统资源监控初始化 =====")
        self.resource_logger.record_system_info()
        
        # 记录训练开始时使用的设备
        self.log_used_devices()
    
    def log_used_devices(self):
        """记录当前训练实际使用的设备"""
        try:
            # 获取当前模型使用的设备
            used_devices = set()
            for layer in self.model.layers:
                for weight in layer.weights:
                    device = weight.device
                    if device:
                        used_devices.add(device)
            
            self.resource_logger.log("\n===== 训练使用的设备 =====")
            for device in used_devices:
                self.resource_logger.log(f"使用设备: {device}")
                if device not in self.resource_logger.used_devices:
                    self.resource_logger.used_devices.append(device)
            self.resource_logger.log("==========================\n")
        except Exception as e:
            self.resource_logger.log(f"记录使用设备时出错: {str(e)}")
    
    def on_train_batch_end(self, batch: int, logs=None) -> None:
        self.batch_counter += 1
        # 每N个批次记录一次资源使用情况，而不仅仅是更新峰值
        if self.batch_counter % self.interval == 0:
            self.resource_logger.update_memory_peaks(batch=batch)
            self.log_batch_resource_usage(batch)
            
    def log_batch_resource_usage(self, batch):
        """记录当前批次的资源使用情况"""
        self.resource_logger.log(f"\n===== Batch {batch} 资源使用统计 =====")
        
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
        self.sample_batch = sample_batch  # (x, y)元组
        self.strategy = strategy  # 传入分布式策略
        self.resource_logger = resource_logger or ResourceLogger()
        self.loss_fn = self.model.loss
        if isinstance(self.loss_fn, str):
            self.loss_fn = keras.losses.get(self.loss_fn)
        self.x_batch, self.y_batch = sample_batch
        # 确保监控批次适配分布式训练
        self.dist_dataset = self._prepare_distributed_dataset()

    def _prepare_distributed_dataset(self) -> tf.data.Dataset:
        """将监控数据转换为分布式数据集"""
        dataset = tf.data.Dataset.from_tensor_slices((self.x_batch, self.y_batch))
        dataset = dataset.batch(len(self.x_batch))  # 单批次
        return self.strategy.experimental_distribute_dataset(dataset)

    def _compute_gradients(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """在副本上下文中计算梯度的函数"""
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
    """RNN基础类，支持动态填充及多GPU训练，整合代码一和代码二功能"""
    def __init__(self,
                 batch_size: int = 5,  # 代码二默认值
                 n_epochs: int = 1400,
                 input_size: int = 1800,
                 set_sample_weights: bool = True,
                 feature_scaling: bool = True,
                 output_size: int = 4,  # 保留代码一的output_size参数
                 training_reshape: bool = False,  # 代码二参数
                 validation_split: float = 0.2):  # 代码二参数
        self.classes_: Optional[List[int]] = None
        self.padding_class: Optional[int] = None  # 填充类别标记
        self.max_seq_len: Optional[int] = None    # 训练时的最大序列长度
        self.input_size = input_size
        self.original_lengths: List[int] = []     # 存储训练样本原始长度
        self.output_size = output_size  # 存储动态输出维度

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.padding = "valid"  # 代码二使用valid padding
        self.set_sample_weights = set_sample_weights
        self.feature_scaling = feature_scaling
        self.model: Optional[keras.Model] = None
        self.weights_: Optional[List[np.ndarray]] = None
        self.model_save_path = "./model_checkpoints"
        self.nan_detector = NaNDetector(verbose=True)
        self.strategy: Optional[tf.distribute.Strategy] = None
        self.resource_logger = ResourceLogger()
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # 代码二特有参数
        self.training_reshape = training_reshape
        self.validation_split = validation_split
        self.ghost_dim = 2  # 代码二特有参数
        self.feature_dim = 1  # 代码二特有参数
        self.fold_index = 0  # 代码二用于多折训练的标识

    def fit(self, X: Union[List[np.ndarray], np.ndarray], y: Union[List[np.ndarray], np.ndarray], fold_index=0) -> None:
        """训练模型，整合代码二的功能"""
        # 代码二功能：接收折数参数
        self.fold_index = fold_index
        
        self.nan_detector = NaNDetector(verbose=True)
        training_start_time = time.time()
        try:
            self.resource_logger.log("="*60)
            self.resource_logger.log("【训练数据信息】")
            self.resource_logger.log(f"原始X类型: {type(X)}, 长度: {len(X) if isinstance(X, (list, np.ndarray)) else 'N/A'}")
            self.resource_logger.log(f"原始y类型: {type(y)}, 长度: {len(y) if isinstance(y, (list, np.ndarray)) else 'N/A'}")
            self.resource_logger.log("="*60)
            
            # 代码二功能：确保X和y是列表
            if not isinstance(X, list):
                X = [X]
                y = [y]
            
            # 提取嵌套样本
            if isinstance(X, list) and len(X) == 1 and isinstance(X[0], (list, np.ndarray)):
                X = X[0]
                self.nan_detector.log_process("提取嵌套X样本")
            if isinstance(y, list) and len(y) == 1 and isinstance(y[0], (list, np.ndarray)):
                y = y[0]
                self.nan_detector.log_process("提取嵌套y样本")
            
            # 检查NaN
            self.nan_detector.check_nan(X, "原始X数据")
            self.nan_detector.check_nan(y, "原始y数据")
            
            # 处理样本，包含代码二的样本修复逻辑
            shapes = []
            valid_samples = []
            valid_labels = []
            
            for i, (x_item, y_item) in enumerate(zip(X, y)):
                try:
                    if isinstance(x_item, list):
                        try:
                            x_arr = np.array(x_item, dtype='float32')
                        except ValueError:
                            processed = []
                            for elem in x_item:
                                if isinstance(elem, list):
                                    elem = np.array(elem, dtype='float32')
                                processed.append(elem)
                            x_arr = np.array(processed, dtype='float32')
                    
                    elif isinstance(x_item, np.ndarray):
                        x_arr = x_item.astype('float32')
                    else:
                        raise ValueError(f"不支持的特征数据类型: {type(x_item)}")

                    if x_arr.ndim == 1:
                        x_arr = x_arr.reshape(-1, 1)
                    elif x_arr.ndim > 2:
                        x_arr = x_arr.reshape(x_arr.shape[0], -1)
                    
                    shapes.append(x_arr.shape)
                    valid_samples.append(x_arr)

                    if y_item is None:
                        y_arr = np.array([0], dtype=int)
                    elif isinstance(y_item, list):
                        cleaned = []
                        for item in y_item:
                            if item is not None:
                                cleaned.append(item)
                        if not cleaned:
                            cleaned = [0]
                        y_arr = np.array(cleaned, dtype=int)
                    elif isinstance(y_item, np.ndarray):
                        y_arr = y_item.astype(int)
                    else:
                        try:
                            y_val = int(y_item)
                            y_arr = np.array([y_val], dtype=int)
                        except:
                            y_arr = np.array([0], dtype=int)
                    
                    valid_labels.append(y_arr)

                except Exception as e:
                    # 代码二功能：样本强制修复逻辑
                    self.resource_logger.log(f"样本 {i} 处理错误 - {str(e)}，尝试强制修复")
                    try:
                        if self.max_seq_len is not None:
                            forced_x = np.full((self.max_seq_len, self.input_size), 
                                              -100.0, dtype='float32')
                        else:
                            forced_x = np.full((100, self.input_size), -100.0, dtype='float32')
                        valid_samples.append(forced_x)
                        valid_labels.append(np.array([0], dtype=int))
                        shapes.append(forced_x.shape)
                        self.resource_logger.log(f"样本 {i} 已强制修复为默认形状")
                    except:
                        self.resource_logger.log(f"样本 {i} 无法修复，已跳过")
                        continue
            
            # 处理空样本情况（代码二功能）
            if not valid_samples:
                self.resource_logger.log("所有样本都无效，使用默认数据继续")
                self.max_seq_len = 100
                self.feature_dim = self.input_size
                valid_samples = [np.full((self.max_seq_len, self.feature_dim), -100.0, dtype='float32')]
                valid_labels = [np.zeros(self.max_seq_len, dtype=int)]
            else:
                self.max_seq_len = max(shape[0] for shape in shapes)
                self.feature_dim = self.input_size
                self.resource_logger.log(f"目标形状: 时间步={self.max_seq_len}, 特征维度={self.feature_dim}")
            
            # 处理X
            processed_X = []
            for x_arr in valid_samples:
                if x_arr.shape[0] < self.max_seq_len:
                    pad_length = self.max_seq_len - x_arr.shape[0]
                    x_padded = np.pad(x_arr, 
                                     pad_width=((0, pad_length), (0, 0)),
                                     mode='constant', 
                                     constant_values=-100.0)  # 代码二使用-100.0作为填充值
                else:
                    x_padded = x_arr[:self.max_seq_len, :]

                if x_padded.shape[1] < self.feature_dim:
                    pad_feat = self.feature_dim - x_padded.shape[1]
                    x_padded = np.pad(x_padded,
                                     pad_width=((0, 0), (0, pad_feat)),
                                     mode='constant',
                                     constant_values=-100.0)
                else:
                    x_padded = x_padded[:, :self.feature_dim]

                processed_X.append(x_padded.reshape(1, self.max_seq_len, self.feature_dim, 1))

            X = np.concatenate(processed_X, axis=0)
            self.resource_logger.log(f"处理后特征形状: {X.shape}")
            
            # 处理y
            processed_y = []
            for y_arr in valid_labels:
                if y_arr.ndim > 1:
                    y_arr = y_arr.flatten()
                
                if len(y_arr) < self.max_seq_len:
                    pad_value = self.padding_class if self.padding_class is not None else 0
                    y_padded = np.pad(y_arr,
                                     pad_width=(0, self.max_seq_len - len(y_arr)),
                                     mode='constant',
                                     constant_values=pad_value)
                else:
                    y_padded = y_arr[:self.max_seq_len]
                
                processed_y.append(y_padded.reshape(1, -1))

            y = np.concatenate(processed_y, axis=0)
            self.resource_logger.log(f"处理后标签形状: {y.shape}")
            
            X = X.astype('float32')
            y = y.astype('float32')

            self.classes_ = list(set(np.ravel(y)))
            if self.padding_class is None or self.padding_class not in self.classes_:
                self.padding_class = len(self.classes_) if len(self.classes_) > 0 else 0

            # 代码二功能：动态调整验证集比例
            num_samples = X.shape[0]
            actual_validation_split = self.validation_split
            
            if num_samples < 5:
                self.resource_logger.log(f"样本数量较少 ({num_samples}个)，自动调整验证集比例")
                if num_samples == 1:
                    actual_validation_split = 0.0
                else:
                    actual_validation_split = max(1/num_samples, min(0.1, self.validation_split))
                self.resource_logger.log(f"调整后验证集比例: {actual_validation_split:.2f}")

            # 确定输出维度（使用初始化时传入的output_size，保留代码一的维度）
            output_size = self.output_size
            
            # 分布式训练设置（保留代码一的功能）
            try:
                self.strategy = tf.distribute.MirroredStrategy()
                self.resource_logger.log(f"已检测到 {self.strategy.num_replicas_in_sync} 个GPU，将用于分布式训练")
            except tf.errors.UnavailableError:
                self.resource_logger.log("无法初始化分布式策略，将使用单设备训练")
                self.strategy = tf.distribute.get_strategy()  # 默认策略
            
            with self.strategy.scope():
                # 构建模型（采用代码二的模型构建方式）
                self._build_model(output_size)
            
            self.weights_ = copy.deepcopy(self.model.get_weights())
            self.resource_logger.log("\n模型初始化完成，结构如下：")
            self.model.summary(print_fn=self.resource_logger.log)

            # 准备监控批次
            monitor_batch_size = min(self.batch_size, X.shape[0])
            monitor_x = X[:monitor_batch_size]
            monitor_y = y[:monitor_batch_size]
            monitor_batch = (monitor_x, monitor_y)
            self.resource_logger.log(f"监控批次形状 - X: {monitor_x.shape}, y: {monitor_y.shape}")
            self.nan_detector.check_nan(monitor_x, "监控批次X数据")
            self.nan_detector.check_nan(monitor_y, "监控批次y数据")

            # 验证集设置
            use_validation = actual_validation_split > 0
            monitor_loss = 'val_loss' if use_validation else 'loss'
            monitor_acc = 'val_accuracy' if use_validation else 'accuracy'
            self.resource_logger.log(f"使用验证集: {use_validation}, 验证比例: {actual_validation_split}")

            # 代码二功能：创建训练结果目录
            os.makedirs("training_results", exist_ok=True)
            
            # 代码二功能：配置CSV日志回调
            csv_logger = CSVLogger(
                f"training_results/fold_{self.fold_index + 1}_metrics.csv",
                append=False
            )

            # 回调函数（整合两者的回调）
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
                GPUUsageMonitor(interval=10, resource_logger=self.resource_logger),
                csv_logger  # 代码二的CSV日志回调
            ]

            # 样本权重（填充部分权重为0）
            sample_weights: Optional[np.ndarray] = None
            if self.set_sample_weights and y.size > 0:
                sample_weights = self._get_samples_weights(y)
                sample_weights = np.clip(sample_weights, 0.0, 10.0)
                self.resource_logger.log(f"样本权重范围: [{np.min(sample_weights):.4f}, {np.max(sample_weights):.4f}]")
                self.nan_detector.check_nan(sample_weights, "样本权重")

            # 开始训练
            self.resource_logger.log(f"\n【开始训练】样本数: {X.shape[0]}, 批次大小: {self.batch_size}, "
                                   f"使用设备数量: {self.strategy.num_replicas_in_sync}")
            
            # 添加分布式训练的稳定性处理
            try:
                history = self.model.fit(
                    x=X,
                    y=y,
                    epochs=self.n_epochs,
                    verbose=1,
                    batch_size=self.batch_size,
                    validation_split=actual_validation_split,
                    shuffle=True,
                    sample_weight=sample_weights,
                    callbacks=model_callbacks
                )
            except tf.errors.CancelledError as e:
                self.resource_logger.log(f"训练过程中发生分布式通信错误: {str(e)}")
                self.resource_logger.log("尝试使用单设备模式重新训练...")
                
                # 切换到单设备模式
                self.strategy = tf.distribute.get_strategy()  # 默认单设备策略
                with self.strategy.scope():
                    self._build_model(output_size)
                    
                # 重新训练
                history = self.model.fit(
                    x=X,
                    y=y,
                    epochs=self.n_epochs,
                    verbose=1,
                    batch_size=self.batch_size,
                    validation_split=actual_validation_split,
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
                "总训练时间": f"{training_time:.2f}秒",
                "使用设备数量": str(self.strategy.num_replicas_in_sync)
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
        """模型预测，返回裁剪填充后的真实音频结果（保留代码一的维度处理）"""
        try:
            if self.max_seq_len is None:
                raise RuntimeError("请先调用fit方法训练模型")
            
            pred_detector = NaNDetector(verbose=True)
            self.resource_logger.log("\n===== 开始预测 =====")
            
            # 记录预测时使用的设备
            self.resource_logger.log("预测使用的设备:")
            try:
                for layer in self.model.layers:
                    for weight in layer.weights:
                        if weight.device:
                            self.resource_logger.log(f"  {weight.name} 在 {weight.device} 上")
            except Exception as e:
                self.resource_logger.log(f"记录预测设备时出错: {str(e)}")
            
            # 提取嵌套样本
            if isinstance(X, list) and len(X) == 1 and isinstance(X[0], (list, np.ndarray)):
                X = X[0]
                pred_detector.log_process("预测：提取嵌套样本")
            
            # 转换为数组并记录原始长度
            X_array: List[np.ndarray] = []
            pred_original_lengths = []  # 记录预测样本的原始长度
            for i, sample in enumerate(X):
                if isinstance(sample, list):
                    try:
                        sample_array = np.array(sample, dtype='float32')
                        X_array.append(sample_array)
                        pred_original_lengths.append(sample_array.shape[0])  # 记录原始长度
                        self.resource_logger.log(f"预测样本{i}：已从list转换为数组，形状={sample_array.shape}，原始长度={sample_array.shape[0]}")
                    except ValueError as e:
                        self.resource_logger.log(f"预测样本{i}：转换失败！错误：{e}")
                        raise
                elif isinstance(sample, np.ndarray):
                    X_array.append(sample)
                    pred_original_lengths.append(sample.shape[0])  # 记录原始长度
                    self.resource_logger.log(f"预测样本{i}：已是数组，形状={sample.shape}，原始长度={sample.shape[0]}")
                else:
                    raise TypeError(f"预测样本{i}：类型错误={type(sample)}")
            X = X_array
            pred_detector.check_nan(X, "预测：转换为数组后")
            self.resource_logger.log(f"预测样本数量: {len(X)}, 原始长度列表: {pred_original_lengths}")
            
            # 统一样本维度
            X = [
                np.expand_dims(sample, axis=-1) if sample.ndim == 1 
                else np.squeeze(sample, axis=-1) if (sample.ndim == 3 and sample.shape[-1] == 1)
                else sample 
                for sample in X
            ]
            pred_detector.check_nan(X, "预测：统一维度后")
            self.resource_logger.log(f"统一维度后样本形状列表: {[s.shape for s in X]}")
            
            # 填充到训练时的最大序列长度
            X_padded: List[np.ndarray] = []
            for sample in X:
                if sample.ndim != 2:
                    raise ValueError(f"预测样本必须是2维数组，实际形状: {sample.shape}")
                
                seq_len, feat_dim = sample.shape
                # 统一特征维度
                if feat_dim != self.input_size:
                    if feat_dim > self.input_size:
                        sample = sample[:, :self.input_size]
                    else:
                        sample = np.pad(sample, ((0,0), (0, self.input_size-feat_dim)), mode='constant')
                
                # 填充窗口数维度（与训练时一致：右填-100.0，代码二的填充值）
                padded = keras.preprocessing.sequence.pad_sequences(
                    sample.T,
                    maxlen=self.max_seq_len,  # 使用训练时的最大长度
                    padding='post',           # 与训练时一致
                    value=-100.0,             # 代码二使用-100.0
                    dtype='float32'
                ).T
                X_padded.append(padded)
            
            # 转换为数组并添加通道维度
            X = np.array(X_padded, dtype='float32')
            if X.ndim == 3:
                X = np.expand_dims(X, axis=-1)
            pred_detector.check_nan(X, "预测：填充后")
            self.resource_logger.log(f"填充后X形状: {X.shape}")
            
            # 标准化（与训练时一致）
            non_pad_mask = X != -100.0  # 对应代码二的填充值
            if np.any(non_pad_mask):
                mean_val = np.mean(X[non_pad_mask])
                X[~non_pad_mask] = mean_val
                mean = np.mean(X[non_pad_mask])
                std = np.std(X[non_pad_mask])
                X[non_pad_mask] = (X[non_pad_mask] - mean) / (std + 1e-8)
                pred_detector.check_nan(X, "预测：标准化后")
            
            # 代码二功能：特征缩放
            if self.feature_scaling:
                X = (X + 1.0) * 100
            
            # 模型预测（先获取概率再求标签）
            self.resource_logger.log(f"预测输入形状: {X.shape}")
            assert self.model is not None, "模型未初始化，请先训练模型"
            
            # 预测时监控资源使用
            start_time = time.time()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024**3)
            
            y_pred_proba = self.model.predict(X, verbose=0)  # 概率形状：(样本数, 窗口数, 类别数)
            y_pred = y_pred_proba.argmax(axis=-1)  # 取概率最大的标签
            
            # 记录预测资源使用
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024**3)
            self.resource_logger.log(f"预测耗时: {end_time - start_time:.2f}秒")
            self.resource_logger.log(f"预测内存使用变化: {start_memory:.2f} GB -> {end_memory:.2f} GB")

            # 打印预测概率与标签（前3个样本的前5个窗口）
            for i in range(min(3, len(y_pred))):
                self.resource_logger.log(f"样本{i}预测详情：")
                for j in range(min(5, y_pred.shape[1])):
                    probs = y_pred_proba[i, j]  # 该窗口的所有类别概率
                    pred_label = y_pred[i, j]   # 该窗口的预测标签
                    self.resource_logger.log(f"  窗口{j} - 概率: {[round(p, 3) for p in probs]} → 预测标签: {pred_label}")
            
            self.resource_logger.log(f"模型原始预测输出形状: {y_pred.shape}")
            
            # 根据原始长度裁剪，去除填充部分
            trimmed_preds = []
            for i in range(len(y_pred)):
                real_len = pred_original_lengths[i]  # 使用预测样本的原始长度
                trimmed = y_pred[i, :real_len]  # 仅保留真实音频部分
                
                # 打印裁剪前后的标签对比
                self.resource_logger.log(f"样本{i} - 裁剪前标签（最后5个窗口，可能含填充）: {y_pred[i, -5:]}")
                self.resource_logger.log(f"样本{i} - 裁剪后标签（实际窗口，长度{real_len}）: {trimmed[-5:] if len(trimmed)>=5 else trimmed}")
                
                trimmed_preds.append(trimmed)
            
            # 聚合序列结果（每个样本返回一个标签）
            if aggregate:
                self.resource_logger.log("开始聚合预测结果...")
                aggregated = []
                for i, seq in enumerate(trimmed_preds):
                    self.resource_logger.log(f"样本{i}聚合前序列形状: {seq.shape}, 序列内容: {seq[:5]}...")  # 只显示前5个元素
                    
                    if len(seq) == 0:
                        # 空序列处理（使用默认类别0）
                        self.resource_logger.log(f"样本{i}是为空序列，使用默认类别0")
                        aggregated.append(0)
                        continue
                    
                    # 取众数作为样本标签（处理多时间步预测）
                    counts = np.bincount(seq)
                    most_common = np.argmax(counts)
                    aggregated.append(most_common)
                    self.resource_logger.log(f"样本{i}聚合结果: {most_common}, 众数统计: {counts}")
                
                # 转换为一维数组并确保形状正确
                result = np.array(aggregated, dtype=int)
                self.resource_logger.log(f"聚合后结果列表: {aggregated}")
                self.resource_logger.log(f"聚合后NumPy数组形状: {result.shape}")
                
                # 确保结果是一维数组
                if result.ndim != 1:
                    self.resource_logger.log(f"警告：聚合结果不是一维数组，尝试重塑，原始形状: {result.shape}")
                    result = result.flatten()
                    self.resource_logger.log(f"重塑后形状: {result.shape}")
                
                # 最终检查
                if len(result) == 0:
                    self.resource_logger.log("警告：聚合结果为空！")
                elif len(result) != len(trimmed_preds):
                    self.resource_logger.log(f"警告：聚合结果数量与输入样本数量不匹配！输入: {len(trimmed_preds)}, 输出: {len(result)}")
                
                self.resource_logger.log(f"预测完成，聚合后结果形状: {result.shape} (一维数组)")
                return result
            else:
                # 直接返回所有窗口的预测结果，展平为一维数组以匹配标签形状
                flat_predictions = np.concatenate(trimmed_preds) if trimmed_preds else np.array([])
                self.resource_logger.log(f"预测完成，窗口级结果总长度: {len(flat_predictions)}")
                self.resource_logger.log(f"窗口级结果形状: {flat_predictions.shape}")
                return flat_predictions

        except Exception as e:
            self.resource_logger.log(f"预测过程出错: {str(e)}")
            traceback.print_exc()
            raise

    def predict_proba(self, X: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        """预测概率，返回裁剪填充后的真实音频结果（保留代码一的维度处理）"""
        try:
            if self.max_seq_len is None:
                raise RuntimeError("请先调用fit方法训练模型")
            
            pred_detector = NaNDetector(verbose=True)
            self.resource_logger.log("\n===== 开始预测概率 =====")
            
            # 提取嵌套样本
            if isinstance(X, list) and len(X) == 1 and isinstance(X[0], (list, np.ndarray)):
                X = X[0]
                pred_detector.log_process("预测概率：提取嵌套样本")
            
            # 转换为数组并记录原始长度
            X_array: List[np.ndarray] = []
            pred_original_lengths = []  # 记录预测样本的原始长度
            for i, sample in enumerate(X):
                if isinstance(sample, list):
                    try:
                        sample_array = np.array(sample, dtype='float32')
                        X_array.append(sample_array)
                        pred_original_lengths.append(sample_array.shape[0])
                        self.resource_logger.log(f"预测概率样本{i}：已转换为数组，形状={sample_array.shape}，原始长度={sample_array.shape[0]}")
                    except ValueError as e:
                        self.resource_logger.log(f"预测概率样本{i}：转换失败！错误：{e}")
                        raise
                elif isinstance(sample, np.ndarray):
                    X_array.append(sample)
                    pred_original_lengths.append(sample.shape[0])
                    self.resource_logger.log(f"预测概率样本{i}：已是数组，形状={sample.shape}，原始长度={sample.shape[0]}")
                else:
                    raise TypeError(f"预测概率样本{i}：类型错误={type(sample)}")
            X = X_array
            pred_detector.check_nan(X, "预测概率：转换为数组后")
            
            # 统一样本维度
            X = [
                np.expand_dims(sample, axis=-1) if sample.ndim == 1 
                else np.squeeze(sample, axis=-1) if (sample.ndim == 3 and sample.shape[-1] == 1)
                else sample 
                for sample in X
            ]
            pred_detector.check_nan(X, "预测概率：统一维度后")
            
            # 填充到训练时的最大序列长度
            X_padded: List[np.ndarray] = []
            for sample in X:
                if sample.ndim != 2:
                    raise ValueError(f"预测概率样本必须是2维数组，实际形状: {sample.shape}")
                
                seq_len, feat_dim = sample.shape
                if feat_dim != self.input_size:
                    if feat_dim > self.input_size:
                        sample = sample[:, :self.input_size]
                    else:
                        sample = np.pad(sample, ((0,0), (0, self.input_size-feat_dim)), mode='constant')
                
                padded = keras.preprocessing.sequence.pad_sequences(
                    sample.T,
                    maxlen=self.max_seq_len,
                    padding='post',
                    value=-100.0,  # 代码二使用-100.0
                    dtype='float32'
                ).T
                X_padded.append(padded)
            
            # 转换为数组并添加通道维度
            X = np.array(X_padded, dtype='float32')
            if X.ndim == 3:
                X = np.expand_dims(X, axis=-1)
            pred_detector.check_nan(X, "预测概率：填充后")
            
            # 标准化
            non_pad_mask = X != -100.0  # 代码二使用-100.0
            if np.any(non_pad_mask):
                mean_val = np.mean(X[non_pad_mask])
                X[~non_pad_mask] = mean_val
                mean = np.mean(X[non_pad_mask])
                std = np.std(X[non_pad_mask])
                X[non_pad_mask] = (X[non_pad_mask] - mean) / (std + 1e-8)
                pred_detector.check_nan(X, "预测概率：标准化后")
            
            # 代码二功能：特征缩放
            if self.feature_scaling:
                X = (X + 1.0) * 100
            
            # 预测概率
            self.resource_logger.log(f"预测概率输入形状: {X.shape}")
            assert self.model is not None, "模型未初始化，请先训练模型"
            y_pred_proba = self.model.predict(X, verbose=0)
            
            # 根据原始长度裁剪，去除填充部分
            trimmed_probs = []
            for i in range(len(y_pred_proba)):
                real_len = pred_original_lengths[i]
                trimmed = y_pred_proba[i, :real_len, :]  # 仅保留真实音频部分的概率
                trimmed_probs.append(trimmed)
            
            self.resource_logger.log(f"预测概率完成，裁剪后结果数量: {len(trimmed_probs)}，均为原始音频长度")
            return trimmed_probs if len(trimmed_probs) > 1 else trimmed_probs[0]
        except Exception as e:
            self.resource_logger.log(f"预测概率过程出错: {str(e)}")
            traceback.print_exc()
            raise

    def _get_samples_weights(self, y: np.ndarray) -> np.ndarray:
        """计算样本权重，填充类别权重为0，使用代码二的实现"""
        unique_classes, counts = np.unique(np.ravel(y), return_counts=True)
        valid_mask = unique_classes != self.padding_class
        valid_counts = counts[valid_mask]
        
        if len(valid_counts) == 0:
            return np.ones_like(y, dtype=float)
        
        max_count = valid_counts.max()
        class_weight = {}
        for cls, cnt in zip(unique_classes, counts):
            if cls == self.padding_class:
                class_weight[cls] = 0.0
            else:
                class_weight[cls] = max_count / cnt
        
        # 日志输出
        self.resource_logger.log("\n===== 类别权重 =====")
        for cls, cnt, weight in zip(unique_classes, counts, class_weight.values()):
            cls_type = "填充类别" if cls == self.padding_class else "普通类别"
            self.resource_logger.log(f"类别 {cls} ({cls_type}): 样本数={cnt}, 权重={weight:.4f}")
        self.resource_logger.log("====================\n")

        # 生成样本权重矩阵
        sample_weight = np.zeros_like(y, dtype=float)
        for class_num, weight in class_weight.items():
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
    """DeepSound模型，继承自RNN基础类，采用代码二的模型构建方式"""
    def __init__(self,
                 batch_size: int = 5,  # 代码二默认值
                 input_size: int = 1800,
                 output_size: int = 5,  # 代码二默认值
                 n_epochs: int = 1500,  # 代码二默认值
                 training_reshape: bool = True,  # 代码二默认值
                 set_sample_weights: bool = True,
                 feature_scaling: bool = True,
                 validation_split: float = 0.2):  # 代码二参数
        super().__init__(
            batch_size=batch_size,
            n_epochs=n_epochs,
            input_size=input_size,
            set_sample_weights=set_sample_weights,
            feature_scaling=feature_scaling,
            output_size=output_size,
            training_reshape=training_reshape,
            validation_split=validation_split
        )
        
        # 初始化时不构建模型，在fit方法中构建（代码二思路）

    def _build_model(self, output_size):
        """构建模型结构，采用代码二的模型参数配置和构建方式"""
        # 打印模型输出维度
        self.resource_logger.log(f"模型构建 - 输出维度output_size={output_size}，最大序列长度max_seq_len={self.max_seq_len}")
        
        # 使用代码二的layers_config参数
        layers_config = [
            (32, 18, 3, activations.relu),
            (32, 9, 1, activations.relu),
            (128, 3, 1, activations.relu)
        ]

        # CNN子网络 - 用于特征提取
        cnn = Sequential(name='cnn_subnetwork')
        cnn.add(layers.Rescaling(scale=1./255, name='input_rescaling'))  # 代码二使用的缩放

        for ix_l, layer in enumerate(layers_config):
            # 第一个卷积块
            cnn.add(layers.Conv1D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=layer[3],  # 代码二直接在这里设置激活函数
                padding=self.padding,
                data_format=self.data_format,
                kernel_initializer=HeUniform(),
                name=f'conv1d_{ix_l*2 + 1}'
            ))

            # 第二个卷积块
            cnn.add(layers.Conv1D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=layer[3],  # 代码二直接在这里设置激活函数
                padding=self.padding,
                data_format=self.data_format,
                kernel_initializer=HeUniform(),
                name=f'conv1d_{ix_l*2 + 2}'
            ))

            # 除最后一层外添加Dropout
            if ix_l < (len(layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.2, name=f'dropout_{ix_l + 1}'))  # 代码二使用0.2

        cnn.add(layers.MaxPooling1D(4, name='max_pooling1d'))
        cnn.add(layers.Flatten(name='flatten'))
        cnn.add(layers.Dropout(rate=0.2, name='cnn_output_dropout'))  # 代码二使用0.2

        # 使用代码二的FFN参数
        ffn = Sequential(name='ffn_subnetwork')
        ffn.add(layers.Dense(256, activation=activations.relu, kernel_initializer=HeUniform(), name='ffn_dense_1'))
        ffn.add(layers.Dropout(rate=0.2, name='ffn_dropout_1'))  # 代码二使用0.2
        ffn.add(layers.Dense(128, activation=activations.relu, kernel_initializer=HeUniform(), name='ffn_dense_2'))
        ffn.add(layers.Dropout(rate=0.2, name='ffn_dropout_2'))  # 代码二使用0.2
        ffn.add(layers.Dense(output_size, activation=activations.softmax, name='ffn_output'))

        # 使用代码二的GRU参数
        self.model = Sequential([
            layers.InputLayer(input_shape=(None, self.input_size, 1), name='input1'),  # 代码二支持动态长度
            layers.TimeDistributed(cnn, name='time_distributed_cnn'),
            layers.Bidirectional(
                layers.GRU(128,  # 代码二使用128
                           activation="tanh", 
                           return_sequences=True, 
                           dropout=0.2,  # 代码二使用0.2
                           kernel_initializer=HeUniform()),
                name='bidirectional_gru'
            ),
            layers.TimeDistributed(ffn, name='time_distributed_ffn')
        ])

        # 代码二使用Adagrad优化器
        self.model.compile(
            optimizer=Adagrad(),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['accuracy']
        )
