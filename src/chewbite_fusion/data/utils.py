import os
import numpy as np
import pandas as pd
import logging
from scipy.interpolate import interp1d


logger = logging.getLogger('yaer')  # 新增日志实例


def windows2events(y_pred,
                   window_width=0.5,
                   window_overlap=0.5,
                   no_event_class='no-event'):
    """Convert predictions from window-level to event-level with logging."""
    step = window_width * (1 - window_overlap)
    valid_windows = []
    for i, label in enumerate(y_pred):
        if label == no_event_class:
            continue
        start = i * step
        end = start + window_width
        valid_windows.append({
            "start": start,
            "end": end,
            "label": label
        })
    
    # 记录窗口过滤情况
    total_windows = len(y_pred)
    valid_count = len(valid_windows)
    logger.debug(f"窗口转事件：总窗口数={total_windows}，有效事件窗口数={valid_count}，无事件窗口数={total_windows - valid_count}")
    
    if not valid_windows:
        return pd.DataFrame(columns=["start", "end", "label"])
    
    df_pred = pd.DataFrame(valid_windows)
    merged_df = merge_contiguous(df_pred)
    
    # 记录合并前后事件数
    logger.debug(f"事件合并：合并前={len(df_pred)}个窗口事件，合并后={len(merged_df)}个连续事件")
    return merged_df


def merge_contiguous(df):
    """Merge contiguous events with same label, add detailed logging."""
    if df.empty:
        return df
    
    df = df.sort_values("start").reset_index(drop=True)
    merged = [df.iloc[0].to_dict()]
    merge_count = 0  # 记录合并次数
    
    for _, row in df.iloc[1:].iterrows():
        last = merged[-1]
        if row["label"] == last["label"] and row["start"] <= last["end"]:
            merged[-1]["end"] = max(last["end"], row["end"])
            merge_count += 1
        else:
            merged.append(row.to_dict())
    
    logger.debug(f"合并完成：共合并{merge_count}次，合并后事件类型分布：{merged_df['label'].value_counts().to_dict()}")
    return pd.DataFrame(merged)


def load_imu_data_from_file(filename):
    '''Read IMU data stored from Android app and return a Pandas DataFrame instance.'''
    char = os.path.basename(filename)[0]  # a: accelerometer, g: gyroscope, m:magnetometer

    dt = np.dtype([('timestamp', '>i8'),
                   (char + 'x', '>f4'),
                   (char + 'y', '>f4'),
                   (char + 'z', '>f4')])

    with open(filename, 'rb') as f:
        file_data = np.fromfile(f, dtype=dt).byteswap().newbyteorder()
        df = pd.DataFrame(file_data, columns=file_data.dtype.names)

    df["timestamp"] = df["timestamp"] / 1e9
    df.rename(columns={"timestamp": "timestamp_sec"}, inplace=True)
    df["timestamp_relative"] = df.timestamp_sec - df.timestamp_sec.values[0]

    return df


def resample_imu_signal(data, signal_duration_sec, frequency, interpolation_kind='linear'):
    '''Resample a given signal.'''
    axis_cols = [c for c in data.columns if 'timestamp' not in c]

    sequence_end = int(data.timestamp_relative.max()) + 1
    x_values = np.linspace(0, sequence_end,
                           sequence_end * frequency,
                           endpoint=False)

    df = pd.DataFrame({'timestamp_relative': x_values})
    df = df[df.timestamp_relative <= signal_duration_sec]

    for col in axis_cols:
        interpolator = interp1d(data['timestamp_relative'],
                                data[col],
                                kind=interpolation_kind)

        df[col] = interpolator(df.timestamp_relative.values)

    return df


import traceback
from functools import wraps

class NaNDetector:
    """NaN检测工具，可追踪NaN出现的具体过程"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.process_history = []  # 记录处理过程
    
    def log_process(self, process_name):
        """记录当前处理过程"""
        self.process_history.append(process_name)
        if self.verbose:
            print(f"[处理过程] {process_name}")
    
    def check_nan(self, data, process_name=None):
        """检查数据中是否存在NaN"""
        if process_name:
            self.log_process(process_name)
        
        # 检查numpy数组
        if isinstance(data, np.ndarray):
            if np.isnan(data).any():
                self._raise_nan_error(f" numpy数组中存在NaN，形状: {data.shape}")
        
        # 检查pandas数据框/系列
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            nan_count = data.isna().sum().sum()
            if nan_count > 0:
                self._raise_nan_error(f" pandas对象中存在{nan_count}个NaN")
        
        # 检查列表或嵌套列表
        elif isinstance(data, list):
            try:
                arr = np.array(data)
                if np.isnan(arr).any():
                    self._raise_nan_error(f" 列表中存在NaN，长度: {len(data)}")
            except TypeError:
                pass  # 非数值列表不检查
        
        return data  # 允许链式调用
    
    def _raise_nan_error(self, msg):
        """抛出包含处理历史的NaN错误"""
        error_msg = "\n".join([
            "检测到NaN值！",
            f"出现位置: {msg}",
            "处理过程链: " + " -> ".join(self.process_history)
        ])
        raise ValueError(error_msg)
    
    @staticmethod
    def wrap_process(process_func):
        """装饰器：包装处理函数，自动检测输入输出是否含NaN"""
        @wraps(process_func)
        def wrapper(*args, **kwargs):
            detector = NaNDetector(verbose=False)
            process_name = process_func.__name__
            
            # 检查输入参数
            for i, arg in enumerate(args):
                try:
                    detector.check_nan(arg, process_name=f"{process_name} - 输入参数{i}")
                except ValueError:
                    traceback.print_exc()
                    raise
            
            # 执行处理函数
            result = process_func(*args, **kwargs)
            
            # 检查输出结果
            try:
                detector.check_nan(result, process_name=f"{process_name} - 输出结果")
            except ValueError:
                traceback.print_exc()
                raise
            
            return result
        return wrapper