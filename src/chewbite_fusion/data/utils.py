import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def windows2events(y_pred,
                   window_width=0.5,
                   window_overlap=0.5):
    """ Convert predictions from window-level to event-level.

    Parameters
    ----------
    y_true : tensor or numpy.array[str]
        1D data structure with labels (window-level) for a refence input segment.
    y_pred : tensor or numpy.array[str]
        1D data structure with predictions (window-level) for a refence input segment.
    window_width : float
        Size of window in seconds used to split signals.
    window_overlap : float
        Overlapping proportion between to consecutive windows (0.00 - 1.00).
    no_event_class : str
        Identifier used to represent the absence of an event of interest.

    Returns
    -------
    df_pred : pandas DataFrame instance.
        DataFrame with start, end and label columns.
    """

    window_starts = np.array(list(range(len(y_pred)))) *\
        (window_width - (window_width * window_overlap))
    window_ends = window_starts + window_width

    df_pred = pd.DataFrame({
        "start": window_starts,
        "end": window_ends,
        "label": y_pred
    })

    df_pred = merge_contiguous(df_pred)
    return df_pred


def merge_contiguous(df):
    """ Given a pandas DataFrame with start, end and label columns it will merge
        contiguous equally labeled. """

    for i in df.index[:-1]:
        next_label = df.loc[i + 1].label
        if next_label == df.loc[i].label:
            df.loc[i + 1, "start"] = df.loc[i].start
            df.drop(i, inplace=True)

    return df


def load_imu_data_from_file(filename):
    ''' Read IMU data stored from Android app and return a Pandas DataFrame instance.

    Params
    ------
    filename : str
        Complete path to the file to be loaded.

    Return
    ------
    df : Data-Frame instance.
        Pandas Data-Frame instance with the following 4 columns:
        - timestamp_sec: timestamp in seconds.
        - {a, g, m}x: signal on x axis.
        - {a, g, m}y: signal on y axis.
        - {a, g, m}z: signal on z axis.
    '''

    # Extract first letter from file name.
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
    ''' Resample a given signal.

    Params
    ------
    data : Data-Frame instance.
        Data loaded using load_data_from_file method.

    signal_duration_sec : float.
        Total desired duration in seconds (used in order to short resulting signal).

    frequency : int.
        Target frequency.

    interpolation_kind : str.
        Interpolation method used.

    Return
    ------
    df : Data-Frame instance.
        Pandas Data-Frame instance with interpolated signals.
    '''
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



# 现有代码结束后（如 resample_imu_signal 函数之后）

import numpy as np
import pandas as pd
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