import os
import copy
import time
import numpy as np
import tensorflow as tf
import psutil
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from chewbite_fusion.data.utils import NaNDetector
import traceback


# 修复：将check_input_data定义为模块级函数，确保全局可访问
def check_input_data(data, resource_logger, batch_num):
    """检查输入数据中是否有极端值或NaN"""
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)
    mean_val = tf.reduce_mean(data)
    std_val = tf.math.reduce_std(data)
    has_nan = tf.reduce_any(tf.math.is_nan(data))
    has_inf = tf.reduce_any(tf.math.is_inf(data))

    # 新增：音频极端值检测（标准化后阈值设为[-5,5]）
    audio_extreme_low = -5.0
    audio_extreme_high = 5.0
    has_extreme = tf.reduce_any((data < audio_extreme_low) | (data > audio_extreme_high))

    # 安全处理批次号（增强版：确保转换为Python原生类型）
    batch_num_val = batch_num
    if isinstance(batch_num_val, tf.Tensor):
        # 尝试获取静态值
        static_val = tf.get_static_value(batch_num_val)
        if static_val is not None:
            batch_num_val = static_val
        else:
            # 尝试在Eager模式下转换
            try:
                if tf.executing_eagerly():
                    batch_num_val = batch_num_val.numpy()
                else:
                    batch_num_val = f"Tensor({batch_num_val.dtype})"
            except:
                batch_num_val = "不可转换的Tensor"
    
    # 处理可能的numpy类型
    if isinstance(batch_num_val, np.ndarray):
        batch_num_val = batch_num_val.item()  # 转换为Python标量
    
    # 确保批次号是整数或字符串
    try:
        batch_num_int = int(batch_num_val) if not isinstance(batch_num_val, str) else batch_num_val
    except (ValueError, TypeError):
        batch_num_int = str(batch_num_val)

    # 修复：改进tensor_to_str函数，使用numpy处理避免tf.strings.format问题
    def tensor_to_str(tensor, fmt):
        # 调试信息：打印格式化字符串和张量值
        if tf.executing_eagerly():
            tensor_np = tensor.numpy()
            print(f"格式化字符串: {fmt}, 张量值: {tensor_np}")  # 调试用
            return fmt.format(tensor_np.item())  # 转为标量后格式化
        else:
            # Graph模式下使用兼容方式
            return f"[{tensor.dtype}]"  # 仅返回类型信息，避免格式化错误
    
    # 使用修复后的tensor_to_str构建日志
    log_parts = [
        f"输入数据检查 - 批次 {batch_num_int}:\n",
        f"  数据范围: [{tensor_to_str(min_val, '{:.6f}')}, {tensor_to_str(max_val, '{:.6f}')}]\n",
        f"  均值: {tensor_to_str(mean_val, '{:.6f}')}, 标准差: {tensor_to_str(std_val, '{:.6f}')}\n",
        f"  含NaN: {has_nan.numpy() if tf.executing_eagerly() else '待计算'}, 含Inf: {has_inf.numpy() if tf.executing_eagerly() else '待计算'}"
    ]
    
    # 新增：极端值日志记录
    if tf.executing_eagerly():
        extreme_count = tf.reduce_sum(tf.cast((data < audio_extreme_low) | (data > audio_extreme_high), tf.int32)).numpy()
        total_elements = data.numpy().size
        extreme_ratio = extreme_count / total_elements if total_elements > 0 else 0
        log_parts.append(
            f"  极端值（超出[{audio_extreme_low},{audio_extreme_high}]）: "
            f"数量={extreme_count}, 占比={extreme_ratio:.2%}"
        )
    else:
        log_parts.append(
            f"  极端值（超出[{audio_extreme_low},{audio_extreme_high}]）: 待计算"
        )
    
    # 合并日志消息（兼容Eager和Graph模式）
    if tf.executing_eagerly():
        log_message = ''.join(log_parts)
        resource_logger.log(log_message)
    else:
        # 在graph模式下简化输出
        log_message = tf.strings.join([
            f"输入数据检查 - 批次 {batch_num_int}: ",
            "数据范围: [占位符, 占位符], 均值: 占位符, 标准差: 占位符"
        ])
        tf.print(log_message)
    
    # 如果发现异常值，记录更多信息（仅在Eager模式下处理）
    if tf.executing_eagerly() and (has_nan.numpy() or has_inf.numpy() or has_extreme.numpy()):
        nan_indices = tf.where(tf.math.is_nan(data)) if has_nan.numpy() else []
        inf_indices = tf.where(tf.math.is_inf(data)) if has_inf.numpy() else []
        extreme_indices = tf.where((data < audio_extreme_low) | (data > audio_extreme_high)) if has_extreme.numpy() else []
        
        log_extreme = ""
        if has_nan.numpy():
            log_extreme += f"  NaN位置: {nan_indices.numpy()[:5]} (最多显示5个)\n"
        if has_inf.numpy():
            log_extreme += f"  Inf位置: {inf_indices.numpy()[:5]} (最多显示5个)\n"
        if has_extreme.numpy():
            log_extreme += f"  极端值位置: {extreme_indices.numpy()[:5]} (最多显示5个)\n"
            
        if log_extreme:
            resource_logger.log(log_extreme)
    
    return tf.debugging.check_numerics(data, "输入数据包含NaN/无穷大")


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
                    # 增强梯度范围监控
                    self.resource_logger.log(f"Epoch {epoch} 梯度范数统计: "
                                           f"最大值={np.max(grad_norms):.4f}, "
                                           f"最小值={np.min(grad_norms):.4f}, "
                                           f"平均值={np.mean(grad_norms):.4f}")
                
        except Exception as e:
            self.resource_logger.log(f"梯度监控出错: {str(e)}")
            traceback.print_exc()


# 添加每批次训练的输出检查回调
class BatchOutputMonitor(keras.callbacks.Callback):
    def __init__(self, resource_logger):
        super().__init__()
        self.resource_logger = resource_logger
        
    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # 每10个批次检查一次
            self.resource_logger.log(f"\n===== 批次 {batch} 训练状态 =====")
            self.resource_logger.log(f"损失值: {logs.get('loss', '未知')}")
            self.resource_logger.log(f"准确率: {logs.get('accuracy', '未知')}")
            self.resource_logger.log("===========================")


# 新增：监控conv1d_5权重的回调
class Conv1DWeightMonitor(Callback):
    def __init__(self, resource_logger):
        super().__init__()
        self.resource_logger = resource_logger
        
    def on_batch_end(self, batch, logs=None):
        if batch % 5 == 0:  # 每5个批次检查一次
            try:
                # 查找conv1d_5层，考虑可能在cnn_subnetwork中的情况
                def find_layer(model, name):
                    for layer in model.layers:
                        if layer.name == name:
                            return layer
                        if hasattr(layer, 'layers'):
                            found = find_layer(layer, name)
                            if found:
                                return found
                    return None
                
                conv1d_5 = find_layer(self.model, 'conv1d_5')
                if conv1d_5 is None:
                    self.resource_logger.log(f"conv1d_5权重监控 - 批次 {batch}: 未找到conv1d_5层")
                    return
                    
                kernel = conv1d_5.kernel
                bias = conv1d_5.bias
                
                # 记录权重范围（修复：兼容计算图模式）
                def get_val(tensor):
                    return tensor.numpy() if tf.executing_eagerly() else "无法获取"
                
                kernel_min = get_val(tf.reduce_min(kernel))
                kernel_max = get_val(tf.reduce_max(kernel))
                bias_min = get_val(tf.reduce_min(bias))
                bias_max = get_val(tf.reduce_max(bias))
                
                # 检查是否有NaN
                has_nan_kernel = get_val(tf.reduce_any(tf.math.is_nan(kernel)))
                has_nan_bias = get_val(tf.reduce_any(tf.math.is_nan(bias)))
                
                self.resource_logger.log(
                    f"conv1d_5 权重监控 - 批次 {batch}:\n"
                    f"  权重范围: [{kernel_min:.6f}, {kernel_max:.6f}]\n"
                    f"  偏置范围: [{bias_min:.6f}, {bias_max:.6f}]\n"
                    f"  权重含NaN: {has_nan_kernel}, 偏置含NaN: {has_nan_bias}"
                )
            except Exception as e:
                self.resource_logger.log(f"conv1d_5权重监控出错: {str(e)}")


# 新增：监控conv1d_5梯度的回调
class Conv1DGradientMonitor(Callback):
    def __init__(self, resource_logger, model, strategy):
        super().__init__()
        self.resource_logger = resource_logger
        self.model = model
        self.strategy = strategy
        self.loss_fn = self.model.loss
        if isinstance(self.loss_fn, str):
            self.loss_fn = keras.losses.get(self.loss_fn)
            
    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # 每10个批次检查一次
            try:
                # 获取最后一个批次的数据
                batch_data = self.model._get_last_batch()
                if batch_data is None:
                    return
                    
                # 处理可能包含样本权重的批次数据
                if len(batch_data) == 3:
                    x_batch, y_batch, sample_weight = batch_data
                else:
                    x_batch, y_batch = batch_data
                    sample_weight = None
                
                # 计算梯度
                with tf.GradientTape() as tape:
                    y_pred = self.model(x_batch, training=True)
                    loss = self.loss_fn(y_batch, y_pred, sample_weight=sample_weight)
                
                # 查找conv1d_5层
                def find_layer(model, name):
                    for layer in model.layers:
                        if layer.name == name:
                            return layer
                        if hasattr(layer, 'layers'):
                            found = find_layer(layer, name)
                            if found:
                                return found
                    return None
                
                conv1d_5 = find_layer(self.model, 'conv1d_5')
                if conv1d_5 is None:
                    self.resource_logger.log(f"conv1d_5梯度监控 - 批次 {batch}: 未找到conv1d_5层")
                    return
                
                # 获取conv1d_5的梯度
                grads = tape.gradient(loss, conv1d_5.trainable_weights)
                
                # 分析梯度（修复：兼容计算图模式）
                def get_val(tensor):
                    return tensor.numpy() if tf.executing_eagerly() else "无法获取"
                
                for grad, var in zip(grads, conv1d_5.trainable_weights):
                    if grad is not None:
                        grad_min = get_val(tf.reduce_min(grad))
                        grad_max = get_val(tf.reduce_max(grad))
                        has_nan = get_val(tf.reduce_any(tf.math.is_nan(grad)))
                        has_inf = get_val(tf.reduce_any(tf.math.is_inf(grad)))
                        
                        self.resource_logger.log(
                            f"conv1d_5 梯度监控 - 批次 {batch} - {var.name}:\n"
                            f"  梯度范围: [{grad_min:.6f}, {grad_max:.6f}]\n"
                            f"  含NaN: {has_nan}, 含Inf: {has_inf}"
                        )
            except Exception as e:
                self.resource_logger.log(f"conv1d_5梯度监控出错: {str(e)}")


class DeepSoundBaseRNN:
    """RNN基础类，支持动态填充及多GPU训练"""
    def __init__(self,
                 batch_size: int = 8,
                 n_epochs: int = 1400,
                 input_size: int = 1800,
                 set_sample_weights: bool = True,
                 feature_scaling: bool = True,
                 output_size: int = 4):  # 修复：添加output_size参数
        self.classes_: Optional[List[int]] = None
        self.padding_class: Optional[int] = None  # 填充类别标记
        self.max_seq_len: Optional[int] = None    # 训练时的最大序列长度
        self.input_size = input_size
        self.original_lengths: List[int] = []     # 存储训练样本原始长度
        self.output_size = output_size  # 修复：存储output_size

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.padding = "same"
        self.set_sample_weights = set_sample_weights
        self.feature_scaling = feature_scaling
        self.model: Optional[keras.Model] = None
        self.weights_: Optional[List[np.ndarray]] = None
        self.model_save_path = "./model_checkpoints"
        self.nan_detector = NaNDetector(verbose=True)
        self.strategy: Optional[tf.distribute.Strategy] = None
        self.resource_logger = ResourceLogger()
        os.makedirs(self.model_save_path, exist_ok=True)
        self.last_batch = None  # 用于存储最后一个批次的数据，供梯度监控使用

    def _build_model(self, max_seq_len: int, output_size: int = 4) -> keras.Model:
        """构建模型结构，优化维度转换和正则化（减少显存占用）"""
        # 核心修改1：减少CNN卷积核数量，降低特征维度
        layers_config = [
            (16, 18, 3, activations.relu),  # 原32 -> 16
            (16, 9, 1, activations.relu),   # 原32 -> 16
            (32, 3, 1, activations.relu)    # 原64 -> 32
        ]

        # CNN子网络 - 用于特征提取
        cnn = Sequential(name='cnn_subnetwork')
        # 添加Masking层，使用-10作为掩码值
        cnn.add(layers.Masking(mask_value=-10.0, name='masking_layer'))
        cnn.add(layers.Rescaling(scale=1.0, name='input_rescaling'))

        for ix_l, layer in enumerate(layers_config):
            # 第一个卷积块
            # 对conv1d_1层添加额外的输入监控
            layer_name = f'conv1d_{ix_l*2 + 1}'
            if layer_name == 'conv1d_1':
                cnn.add(layers.Lambda(
                    lambda x: [
                        tf.print("conv1d_1输入详细统计: min=", tf.reduce_min(x), 
                                "max=", tf.reduce_max(x), "mean=", tf.reduce_mean(x),
                                "std=", tf.math.reduce_std(x), "nan_count=", tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int32))),
                        x
                    ][1],
                    name='conv1d_1_input_detailed'
                ))
            
            cnn.add(layers.Conv1D(
                layer[0],
                kernel_size=layer[1],
                strides=layer[2],
                activation=None,
                padding=self.padding,
                data_format=self.data_format,
                kernel_initializer=HeUniform(),
                name=layer_name
            ))
            
            # 对conv1d_1层添加额外的输出监控
            if layer_name == 'conv1d_1':
                cnn.add(layers.Lambda(
                    lambda x: [
                        tf.print("conv1d_1输出详细统计: min=", tf.reduce_min(x), 
                                "max=", tf.reduce_max(x), "mean=", tf.reduce_mean(x),
                                "std=", tf.math.reduce_std(x), "nan_count=", tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int32))),
                        x
                    ][1],
                    name='conv1d_1_output_detailed'
                ))
            
            # 添加卷积后监控
            cnn.add(layers.Lambda(
                lambda x: [
                    tf.print(f"{layer_name}输出范围: ", tf.reduce_min(x), tf.reduce_max(x)),
                    tf.debugging.check_numerics(x, f"{layer_name}包含NaN/无穷大"),
                    x
                ][2],
                name=f'{layer_name}_monitor'
            ))
            cnn.add(layers.BatchNormalization(name=f'bn_{ix_l*2 + 1}'))
            # 添加BN后监控
            cnn.add(layers.Lambda(
                lambda x: [
                    tf.print(f"bn_{ix_l*2 + 1}输出范围: ", tf.reduce_min(x), tf.reduce_max(x)),
                    tf.debugging.check_numerics(x, f"bn_{ix_l*2 + 1}包含NaN/无穷大"),
                    x
                ][2],
                name=f'bn_{ix_l*2 + 1}_monitor'
            ))
            cnn.add(layers.Activation(layer[3], name=f'act_{ix_l*2 + 1}'))

            # 第二个卷积块（相同方式添加监控）
            # 重点监控conv1d_5层（ix_l=2时）
            layer_name = f'conv1d_{ix_l*2 + 2}'
            
            # 对conv1d_5层添加额外监控和处理
            if layer_name == 'conv1d_5':
                # 输入监控
                cnn.add(layers.Lambda(
                    lambda x: [
                        tf.print("conv1d_5输入范围:", tf.reduce_min(x), tf.reduce_max(x),
                                "是否含NaN:", tf.reduce_any(tf.math.is_nan(x))),
                        tf.debugging.check_numerics(x, "conv1d_5输入包含NaN/无穷大"),
                        # 添加数值范围限制
                        tf.clip_by_value(x, clip_value_min=-10.0, clip_value_max=10.0)
                    ][2],
                    name='conv1d_5_input_monitor'
                ))
                
                # 添加conv1d_5层（只添加一次）
                cnn.add(layers.Conv1D(
                    layer[0],
                    kernel_size=layer[1],
                    strides=layer[2],
                    activation=None,
                    padding=self.padding,
                    data_format=self.data_format,
                    kernel_initializer=HeUniform(),
                    name=layer_name
                ))
                
                # 输出监控
                cnn.add(layers.Lambda(
                    lambda x: [
                        tf.print("conv1d_5输出范围:", tf.reduce_min(x), tf.reduce_max(x),
                                "是否含NaN:", tf.reduce_any(tf.math.is_nan(x))),
                        tf.debugging.check_numerics(x, "conv1d_5输出包含NaN/无穷大"),
                        # 添加数值范围限制
                        tf.clip_by_value(x, clip_value_min=-10.0, clip_value_max=10.0)
                    ][2],
                    name='conv1d_5_output_monitor'
                ))
            else:
                # 其他卷积层正常添加
                cnn.add(layers.Conv1D(
                    layer[0],
                    kernel_size=layer[1],
                    strides=layer[2],
                    activation=None,
                    padding=self.padding,
                    data_format=self.data_format,
                    kernel_initializer=HeUniform(),
                    name=layer_name
                ))
                
                cnn.add(layers.Lambda(
                    lambda x, ln=layer_name: [
                        tf.print(f"{ln}输出范围: ", tf.reduce_min(x), tf.reduce_max(x)),
                        tf.debugging.check_numerics(x, f"{ln}包含NaN/无穷大"),
                        x
                    ][2],
                    name=f'{layer_name}_monitor'
                ))
            
            cnn.add(layers.BatchNormalization(name=f'bn_{ix_l*2 + 2}'))
            cnn.add(layers.Lambda(
                lambda x: [
                    tf.print(f"bn_{ix_l*2 + 2}输出范围: ", tf.reduce_min(x), tf.reduce_max(x)),
                    tf.debugging.check_numerics(x, f"bn_{ix_l*2 + 2}包含NaN/无穷大"),
                    x
                ][2],
                name=f'bn_{ix_l*2 + 2}_monitor'
            ))
            cnn.add(layers.Activation(layer[3], name=f'act_{ix_l*2 + 2}'))

            # 除最后一层外添加Dropout
            if ix_l < (len(layers_config) - 1):
                cnn.add(layers.Dropout(rate=0.3, name=f'dropout_{ix_l + 1}'))

        cnn.add(layers.MaxPooling1D(4, name='max_pooling1d'))
        cnn.add(layers.Flatten(name='flatten'))
        cnn.add(layers.Dropout(rate=0.2, name='cnn_output_dropout'))

        # 核心修改2：减少FFN维度，降低参数数量
        ffn = Sequential(name='ffn_subnetwork')
        ffn.add(layers.Dense(128, activation=None, kernel_initializer=HeUniform(), name='ffn_dense_1'))  # 原256 -> 128
        ffn.add(layers.BatchNormalization(name='ffn_bn_1'))
        ffn.add(layers.Activation(activations.relu, name='ffn_act_1'))
        ffn.add(layers.Dropout(rate=0.3, name='ffn_dropout_1'))
        
        ffn.add(layers.Dense(64, activation=None, kernel_initializer=HeUniform(), name='ffn_dense_2'))   # 原128 -> 64
        ffn.add(layers.BatchNormalization(name='ffn_bn_2'))
        ffn.add(layers.Activation(activations.relu, name='ffn_act_2'))
        ffn.add(layers.Dropout(rate=0.3, name='ffn_dropout_2'))
        
        ffn.add(layers.Dense(self.output_size, activation=activations.softmax, name='ffn_output'))

        # 核心修改3：修复监控层，移除assert操作以兼容分布式训练
        model = Sequential([
            layers.InputLayer(input_shape=(max_seq_len, self.input_size, 1), name='input1'),
            # 监控输入层 - 增强版：移除assert操作，保留必要的监控
            layers.Lambda(
                lambda x: [
                    tf.print("输入层详细统计: min=", tf.reduce_min(x), "max=", tf.reduce_max(x), 
                            "mean=", tf.reduce_mean(x), "std=", tf.math.reduce_std(x),
                            "nan_count=", tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int32))),
                    tf.debugging.assert_all_finite(x, "输入数据包含NaN/无穷大"),
                    # 移除tf.debugging.assert_less_equal，避免创建Operation对象
                    x
                ][2],
                name='input_monitor'
            ),
            # CNN层及输出监控 - 修正版
            layers.TimeDistributed(cnn, name='time_distributed_cnn'),
            layers.Lambda(
                lambda x: [
                    tf.print("CNN输出形状:", tf.shape(x)),
                    tf.debugging.check_numerics(x, "CNN输出包含NaN/无穷大"),
                    x  # 返回原始张量
                ][2],  # 取列表中第三个元素（即x）作为输出
                name='cnn_output_monitor'
            ),
            # GRU层及输入监控 - 修正版
            layers.Lambda(
                lambda x: [
                    tf.print("GRU输入形状:", tf.shape(x)),
                    x  # 返回原始张量
                ][1],  # 取列表中第二个元素（即x）作为输出
                name='gru_input_monitor'
            ),
            layers.Bidirectional(
                layers.GRU(64,
                           activation="tanh", 
                           return_sequences=True, 
                           dropout=0.3,
                           recurrent_dropout=0.2,
                           kernel_initializer=HeUniform()),
                name='bidirectional_gru'
            ),
            # GRU输出监控 - 修正版
            layers.Lambda(
                lambda x: [
                    tf.print("GRU输出形状:", tf.shape(x)),
                    tf.debugging.check_numerics(x, "GRU输出包含NaN/无穷大"),
                    x  # 返回原始张量
                ][2],  # 取列表中第三个元素（即x）作为输出
                name='gru_output_monitor'
            ),
            # FFN层及输入监控 - 修正版
            layers.Lambda(
                lambda x: [
                    tf.print("FFN输入形状:", tf.shape(x)),
                    x  # 返回原始张量
                ][1],  # 取列表中第二个元素（即x）作为输出
                name='ffn_input_monitor'
            ),
            layers.TimeDistributed(ffn, name='time_distributed_ffn'),
            # FFN输出监控 - 修正版
            layers.Lambda(
                lambda x: [
                    tf.print("FFN输出形状:", tf.shape(x)),
                    tf.debugging.check_numerics(x, "FFN输出包含NaN/无穷大"),
                    x  # 返回原始张量
                ][2],  # 取列表中第三个元素（即x）作为输出
                name='ffn_output_monitor'
            )
        ])

        # 新增：自定义训练步骤以监控批次数据，支持样本权重
        class CustomModel(keras.Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._train_counter = tf.Variable(0, trainable=False, dtype=tf.int64)  # 初始化批次计数器
                self._last_batch = None  # 存储最后一个批次数据
                
            def train_step(self, data):
                # 新增：监控学习率（修复分布式环境下的学习率获取方式）
                current_lr = tf.keras.backend.get_value(self.optimizer.lr)
                tf.print("当前学习率:", current_lr)
                
                # 处理包含样本权重的三元组或普通二元组
                if len(data) == 3:
                    x, y, sample_weight = data
                else:
                    x, y = data
                    sample_weight = None
                
                # 保存批次数据供后续梯度监控（包含样本权重）
                self._last_batch = (x, y, sample_weight) if sample_weight is not None else (x, y)
                
                # 获取批次号（使用numpy值避免Tensor格式化问题）
                train_counter = self._train_counter.numpy() if tf.executing_eagerly() else "graph_mode"
                x = check_input_data(x, self.resource_logger, train_counter)
                
                # 前向传播
                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    # 计算损失时传入样本权重
                    loss = self.compiled_loss(
                        y, y_pred,
                        sample_weight=sample_weight,
                        regularization_losses=self.losses
                    )
                
                # 计算梯度并更新权重
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                
                # 更新指标（传入样本权重）
                self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
                
                # 批次计数器递增
                self._train_counter.assign_add(1)
                
                # 返回指标结果
                return {m.name: m.result() for m in self.metrics}
                
            def _get_last_batch(self):
                return self._last_batch

        # 转换为自定义模型
        custom_model = CustomModel(model.input, model.output)
        custom_model.resource_logger = self.resource_logger
        
        # 优化器配置调整：降低学习率并增强梯度裁剪
        custom_model.compile(
            optimizer=Adam(
                learning_rate=1e-5,  # 降低学习率
                clipnorm=1.0,        # 增强梯度裁剪
                clipvalue=0.5
            ),
            loss='sparse_categorical_crossentropy',
            weighted_metrics=['accuracy']
        )

        return custom_model

    def fit(self, X: Union[List[np.ndarray], np.ndarray], y: Union[List[np.ndarray], np.ndarray]) -> None:
        """训练模型"""
        self.nan_detector = NaNDetector(verbose=True)
        training_start_time = time.time()
        try:
            self.resource_logger.log("="*60)
            self.resource_logger.log("【训练数据信息】")
            self.resource_logger.log(f"原始X类型: {type(X)}, 长度: {len(X) if isinstance(X, (list, np.ndarray)) else 'N/A'}")
            self.resource_logger.log(f"原始y类型: {type(y)}, 长度: {len(y) if isinstance(y, (list, np.ndarray)) else 'N/A'}")
            self.resource_logger.log("="*60)
            
            # 提取嵌套样本
            if isinstance(X, list) and len(X) == 1 and isinstance(X[0], (list, np.ndarray)):
                X = X[0]
                self.nan_detector.log_process("提取嵌套X样本")
            
            # 检查NaN
            self.nan_detector.check_nan(X, "原始X数据")
            
            # 转换为NumPy数组
            self.resource_logger.log("\n===== 转换样本为NumPy数组 =====")
            X_array: List[np.ndarray] = []
            self.original_lengths = []  # 重置并记录原始长度
            for i, sample in enumerate(X):
                if isinstance(sample, list):
                    try:
                        sample_array = np.array(sample, dtype='float32')
                        X_array.append(sample_array)
                        self.original_lengths.append(len(sample_array))  # 记录原始长度
                        self.resource_logger.log(f"样本{i}：已从list转换为数组，形状={sample_array.shape}，原始长度={len(sample_array)}")
                    except ValueError as e:
                        self.resource_logger.log(f"样本{i}：列表转换为数组失败！错误：{e}")
                        raise
                elif isinstance(sample, np.ndarray):
                    X_array.append(sample)
                    self.original_lengths.append(len(sample))  # 记录原始长度
                    self.resource_logger.log(f"样本{i}：已是数组，形状={sample.shape}，原始长度={len(sample)}")
                else:
                    raise TypeError(f"样本{i}：既不是list也不是数组，类型={type(sample)}")
            X = X_array
            self.nan_detector.check_nan(X, "转换为数组后的X")
            self.resource_logger.log("===========================\n")
            
            # 统一样本维度
            X = [
                np.expand_dims(sample, axis=-1) if (isinstance(sample, np.ndarray) and sample.ndim == 1) 
                else np.squeeze(sample, axis=-1) if (isinstance(sample, np.ndarray) and sample.ndim == 3 and sample.shape[-1] == 1)
                else sample 
                for sample in X
            ]
            self.nan_detector.check_nan(X, "统一维度后的X")
            
            # 计算最大序列长度
            if isinstance(X, (list, np.ndarray)):
                self.max_seq_len = max(len(sample) for sample in X) if X else 0
                self.resource_logger.log(f"当前批次最长序列长度（窗口数）: {self.max_seq_len}")
            else:
                raise ValueError("X必须是列表或NumPy数组")

            # 新增：处理并检查标签y的维度和数值
            self.resource_logger.log("\n===== 处理并检查标签y =====")
            # 提取嵌套y样本（与X处理逻辑一致）
            if isinstance(y, list) and len(y) == 1 and isinstance(y[0], (list, np.ndarray)):
                y = y[0]
                self.nan_detector.log_process("提取嵌套y样本")
            
            # 转换y为数组并检查长度
            y_array: List[np.ndarray] = []
            for i, label_seq in enumerate(y):
                if isinstance(label_seq, list):
                    label_array = np.array(label_seq, dtype='int32')  # 标签应为整数
                    y_array.append(label_array)
                elif isinstance(label_seq, np.ndarray):
                    y_array.append(label_seq.astype('int32'))
                else:
                    raise TypeError(f"标签样本{i}类型错误: {type(label_seq)}")
                
                # 检查标签序列长度是否与X样本一致
                x_len = len(X[i])
                y_len = len(y_array[i])
                if x_len != y_len:
                    raise ValueError(
                        f"样本{i}的X与y长度不匹配: X长度={x_len}, y长度={y_len}"
                    )
                self.resource_logger.log(f"样本{i}标签：形状={y_array[i].shape}，类别范围=[{y_array[i].min()}, {y_array[i].max()}]")
                
                # 检查标签是否超出输出维度范围
                if y_array[i].max() >= self.output_size:
                    raise ValueError(
                        f"样本{i}标签值超出范围: 最大标签={y_array[i].max()}, 输出维度={self.output_size}"
                    )
            y = y_array
            self.nan_detector.check_nan(y, "转换后的标签y")  # 检查标签中是否有NaN
            self.resource_logger.log("===========================\n")

            # 同步填充X和y（均在末尾填充）
            target_len = self.max_seq_len
            X_padded: List[np.ndarray] = []
            for sample in X:
                if not isinstance(sample, np.ndarray) or sample.ndim != 2:
                    raise ValueError(f"样本必须是2维数组，实际样本形状: {sample.shape if isinstance(sample, np.ndarray) else type(sample)}")
                
                seq_len, feat_dim = sample.shape
                # 统一特征维度
                if feat_dim != self.input_size:
                    self.resource_logger.log(f"样本特征维度不匹配: 实际{feat_dim}，预期{self.input_size}，自动调整")
                    if feat_dim > self.input_size:
                        sample = sample[:, :self.input_size]
                    else:
                        pad_width = ((0, 0), (0, self.input_size - feat_dim))
                        sample = np.pad(sample, pad_width, mode='constant', constant_values=0.0)
                
                # 填充窗口数维度（右填-10.0，与y填充位置一致）
                padded = keras.preprocessing.sequence.pad_sequences(
                    sample.T,
                    maxlen=target_len,
                    padding='post',
                    value=-10.0,  # 关键修改：使用-10作为填充值
                    dtype='float32'
                ).T
                X_padded.append(padded)
            
            # 转换为数组
            try:
                X = np.array(X_padded, dtype='float32')
            except ValueError as e:
                self.resource_logger.log(f"转换为数组失败: {e}")
                self.resource_logger.log("填充后样本形状:")
                for i, p in enumerate(X_padded):
                    self.resource_logger.log(f"样本{i}: {p.shape}")
                raise
            
            self.nan_detector.check_nan(X, "X填充后的数组")
            
            # 添加通道维度
            if X.ndim == 3:
                X = np.expand_dims(X, axis=-1)
            self.resource_logger.log(f"X填充后形状: {X.shape}")
            self.nan_detector.check_nan(X, "添加通道维度后的X")

            # 处理标签和填充类别
            self.classes_ = list(set(np.concatenate(y))) if y and isinstance(y[0], (list, np.ndarray)) else []
            self.padding_class = max(self.classes_) + 1 if self.classes_ else 0
            self.resource_logger.log(f"检测到的类别: {self.classes_}, 填充类别编号: {self.padding_class}")
            
            # 填充标签（与X同步在末尾填充）
            y_padded = keras.preprocessing.sequence.pad_sequences(
                y,
                maxlen=target_len,
                padding='post',
                value=self.padding_class,
                dtype='int32'
            )
            y = y_padded
            self.resource_logger.log(f"y填充后形状: {y.shape}")
            self.nan_detector.check_nan(y, "y填充后的数据")

            # 处理X的填充值（保留-10作为掩码值，不替换为均值）
            non_pad_mask = X != -10.0  # 关键修改：识别-10作为填充值
            if np.any(non_pad_mask):
                self.resource_logger.log(f"保留填充值-10作为掩码值，不替换为均值")
                self.nan_detector.check_nan(X, "保留填充值后的X")
            else:
                self.resource_logger.log("警告：所有值都是填充值，可能数据异常")

            # 特征标准化 - 确保不影响填充值
            if self.feature_scaling and np.any(non_pad_mask):
                mean = np.mean(X[non_pad_mask])
                std = np.std(X[non_pad_mask])
                # 只标准化非填充值
                X[non_pad_mask] = (X[non_pad_mask] - mean) / (std + 1e-8)
                self.resource_logger.log(f"标准化后X统计: min={np.min(X):.4f}, max={np.max(X):.4f}")
                # 添加更详细的输入数据监控
                self.resource_logger.log(f"标准化后X极端值: 最小={np.min(X):.6f}, 最大={np.max(X):.6f}, 均值={np.mean(X):.6f}, 标准差={np.std(X):.6f}")
                self.resource_logger.log(f"标准化后X是否含Inf: {np.isinf(X).any()}, 位置: {np.where(np.isinf(X))}")
                self.nan_detector.check_nan(X, "标准化后的X")

            # 确定输出维度
            output_size = len(self.classes_) + 1 if self.classes_ else self.output_size  # 使用初始化的output_size
            
            # 分布式训练设置 - 确保在策略作用域内构建模型
            if self.strategy is None:
                self.strategy = tf.distribute.MirroredStrategy()
                self.resource_logger.log(f"已检测到 {self.strategy.num_replicas_in_sync} 个GPU，将用于分布式训练")
            
            with self.strategy.scope():
                self.model = self._build_model(max_seq_len=self.max_seq_len, output_size=output_size)
            
            self.weights_ = copy.deepcopy(self.model.get_weights())
            self.resource_logger.log("\n模型初始化完成（多GPU支持），结构如下：")
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
            use_validation = X.shape[0] >= 5
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
                    layer_names=[
                        'conv1d_1', 'bn_1', 'conv1d_2', 'bn_2',  # 第一层卷积块
                        'conv1d_3', 'bn_3', 'conv1d_4', 'bn_4',  # 第二层卷积块
                        'conv1d_5', 'bn_5', 'conv1d_6', 'bn_6',  # 第三层卷积块
                        'time_distributed_cnn', 'bidirectional_gru', 'time_distributed_ffn'
                    ],
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
                BatchOutputMonitor(resource_logger=self.resource_logger),  # 添加批次监控回调
                Conv1DWeightMonitor(resource_logger=self.resource_logger)  # 新增：conv1d_5权重监控
            ]
            
            # 只有在单GPU训练时添加梯度监控（分布式训练需要特殊处理）
            if self.strategy.num_replicas_in_sync == 1:
                model_callbacks.append(
                    Conv1DGradientMonitor(
                        resource_logger=self.resource_logger,
                        model=self.model,
                        strategy=self.strategy
                    )
                )

            # 样本权重（填充部分权重为0）
            sample_weights: Optional[np.ndarray] = None
            if self.set_sample_weights and y.size > 0:
                sample_weights = self._get_samples_weights(y)
                sample_weights = np.clip(sample_weights, 0.0, 10.0)
                self.resource_logger.log(f"样本权重范围: [{np.min(sample_weights):.4f}, {np.max(sample_weights):.4f}]")
                self.nan_detector.check_nan(sample_weights, "样本权重")

            # 开始训练
            self.resource_logger.log(f"\n【开始训练】样本数: {X.shape[0]}, 批次大小: {self.batch_size}, GPU数量: {self.strategy.num_replicas_in_sync}")
            history = self.model.fit(
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

    def predict(self, X: Union[List[np.ndarray], np.ndarray], aggregate: bool = True) -> Union[List[np.ndarray], np.ndarray]:
        """模型预测，返回裁剪填充后的真实音频结果
        Args:
            X: 输入数据
            aggregate: 是否聚合序列结果（每个样本返回一个标签），True时返回一维数组，False时返回原始序列
        """
        try:
            if self.max_seq_len is None:
                raise RuntimeError("请先调用fit方法训练模型")
            
            pred_detector = NaNDetector(verbose=True)
            self.resource_logger.log("\n===== 开始预测 =====")
            
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
                
                # 填充窗口数维度（与训练时一致：右填-10.0）
                padded = keras.preprocessing.sequence.pad_sequences(
                    sample.T,
                    maxlen=self.max_seq_len,  # 使用训练时的最大长度
                    padding='post',           # 与训练时一致
                    value=-10.0,              # 关键修改：使用-10作为填充值
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
            non_pad_mask = X != -10.0  # 关键修改：识别-10作为填充值
            if np.any(non_pad_mask):
                # 只标准化非填充值
                mean = np.mean(X[non_pad_mask])
                std = np.std(X[non_pad_mask])
                X[non_pad_mask] = (X[non_pad_mask] - mean) / (std + 1e-8)
                pred_detector.check_nan(X, "预测：标准化后")
            
            # 模型预测
            self.resource_logger.log(f"预测输入形状: {X.shape}")
            assert self.model is not None, "模型未初始化，请先训练模型"
            y_pred = self.model.predict(X, verbose=0).argmax(axis=-1)
            self.resource_logger.log(f"模型原始预测输出形状: {y_pred.shape}")
            
            # 根据原始长度裁剪，去除填充部分
            trimmed_preds = []
            for i in range(len(y_pred)):
                real_len = pred_original_lengths[i]  # 使用预测样本的原始长度
                trimmed = y_pred[i, :real_len]  # 仅保留真实音频部分
                trimmed_preds.append(trimmed)
                self.resource_logger.log(f"样本{i}裁剪后形状: {trimmed.shape}, 原始长度: {real_len}")
            
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
                self.resource_logger.log(f"预测完成，裁剪后结果数量: {len(trimmed_preds)}，均为原始音频长度")
                return trimmed_preds if len(trimmed_preds) > 1 else trimmed_preds[0]
        except Exception as e:
            self.resource_logger.log(f"预测过程出错: {str(e)}")
            traceback.print_exc()
            raise

    def predict_proba(self, X: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        """预测概率，返回裁剪填充后的真实音频结果"""
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
                    value=-10.0,  # 关键修改：使用-10作为填充值
                    dtype='float32'
                ).T
                X_padded.append(padded)
            
            # 转换为数组并添加通道维度
            X = np.array(X_padded, dtype='float32')
            if X.ndim == 3:
                X = np.expand_dims(X, axis=-1)
            pred_detector.check_nan(X, "预测概率：填充后")
            
            # 标准化
            non_pad_mask = X != -10.0  # 关键修改：识别-10作为填充值
            if np.any(non_pad_mask):
                # 只标准化非填充值
                mean = np.mean(X[non_pad_mask])
                std = np.std(X[non_pad_mask])
                X[non_pad_mask] = (X[non_pad_mask] - mean) / (std + 1e-8)
                pred_detector.check_nan(X, "预测概率：标准化后")
            
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
        """计算样本权重，填充类别权重为0"""
        unique_classes, counts = np.unique(np.ravel(y), return_counts=True)
        counts = np.maximum(counts, 1)  # 避免除以0
        
        # 计算类别权重（平衡类别）
        class_weight = np.log((counts.max() / counts) + 1.0)
        
        # 填充类别权重设为0（核心：忽略填充部分的影响）
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
    """DeepSound模型，继承自RNN基础类"""
    def __init__(self,
                 batch_size: int = 5,  # 结合实验调整为较小批次
                 input_size: int = 4000,
                 output_size: int = 3,
                 n_epochs: int = 1400,
                 set_sample_weights: bool = True,
                 feature_scaling: bool = True):
        super().__init__(
            batch_size=batch_size,
            n_epochs=n_epochs,
            input_size=input_size,
            set_sample_weights=set_sample_weights,
            feature_scaling=feature_scaling,
            output_size=output_size  # 传递output_size到父类
        )