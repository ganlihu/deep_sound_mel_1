# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import pandas as pd
from scipy import signal
import librosa
import more_itertools

from chewbite_fusion.data.cache_manager import DatasetCache
from chewbite_fusion.data import utils_data_sources as utils
from chewbite_fusion.data.utils import AudioExtremeDetector  # 导入工具类


logger = logging.getLogger(__name__)


def main(data_source_names=['zavalla2022'],
         audio_sampling_frequency=8000,
         movement_sampling_frequency=100,
         window_width=0.5,
         window_overlap=0.5,
         label_overlapping_threshold=0.5,
         filter_noises=True,
         include_movement_magnitudes=False,
         no_event_class_name='no-event',
         filters=None,
         invalidate_cache=True):
    """ Run data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        Parameters
        ----------
        data_source_names : list of str
            List of all data source names to be included in the final dataset.
            At this moment there is one option valid: 'zavalla2022'.
        audio_sampling_frequency : int or float
            Sampling frequency of audio source files.
        movement_sampling_frequency : int or float
            Sampling frequency of IMU source files.
        window_width : float
            Size of window in seconds used to split signals.
        window_overlap : float
            Overlapping proportion between to consecutive windows (0.00 - 1.00).
        label_overlapping_threshold : float
            Minimun threshold to assign a label to frame w.r.t. window width (0.00 - 1.00).
        filter_noises : bool
            Define if parts of original signals which include noises are included.
        include_movement_magnitudes : bool
            Define if magnitudes of IMU data are calculated.
        no_event_class_name : str
            Class name to represent the absense of an event of interest.
        filters : list of tuples.
            List of filters, channels and a flag to indicate if applied to movement signals.
            For example, [(signal.butter(10, 15, 'hp'), [0, 1, 2], True)]
                apply a 15th order high-pass Butterworth filter to acceleromter x, y and z axis.
        invalidate_cache : bool
            Force to update cache.

        Returns
        -------
        X : Dictionary-like object, with data sources as keys.
            Each value represent segments of data, and include all extracted windows.
        y : Dictionary-like object, with data sources as keys.
            Each value represent segments of data, and include labels for each window.
    """
    logger = logging.getLogger(__name__)

    cache = DatasetCache()

    # Try to retrieve elements from cache.
    cache_item = cache.load(
        data_source_names=data_source_names,
        audio_sampling_frequency=audio_sampling_frequency,
        movement_sampling_frequency=movement_sampling_frequency,
        window_width=window_width,
        window_overlap=window_overlap,
        label_overlapping_threshold=label_overlapping_threshold,
        filter_noises=filter_noises,
        include_movement_magnitudes=include_movement_magnitudes,
        no_event_class_name=no_event_class_name,
        filters=filters)

    if cache_item and not invalidate_cache:
        logger.info('*** Retrieving dataset from cache ! ***')
        (X, y) = cache_item

        return X, y

    logger.info('*** Creating dataset from scratch ! ***')
    available_datasets = utils.list_datasets()

    for data_source_name in data_source_names:
        assert data_source_name in available_datasets, \
            f'Provided data source name {data_source_name} not available.'

    assert (audio_sampling_frequency * window_width) % 5 == 0, \
        '''Incompatible audio sampling frequency and window width
           (Validation condition: audio_sampling_frequency * window_width) % 5).'''

    assert (audio_sampling_frequency * window_width * (1 - window_overlap)) % 5 == 0, \
        '''Incompatible audio sampling frequency and window overlap
           (Validation condition:
           audio_sampling_frequency * window_width * (1 - window_overlap)) % 5).'''

    assert (movement_sampling_frequency * window_width) % 5 == 0, \
        '''Incompatible movement sampling frequency and window width
           (Validation condition: movement_sampling_frequency * window_width) % 5).'''

    assert (movement_sampling_frequency * window_width * (1 - window_overlap)) % 5 == 0, \
        '''Incompatible movement sampling frequency and window overlap
           (Validation condition:
           movement_sampling_frequency * window_width * (1 - window_overlap)) % 5).'''

    X = {}
    y = {}

    for dataset in data_source_names:
        segment_files = utils.get_files_in_dataset(available_datasets[dataset])

        X_dataset_segments = {}
        y_dataset_segments = {}
        for segment in segment_files:
            segment_name = os.path.basename(segment[0]).split('.')[0]
            
            # 新增日志：打印当前片段的长度和包含的文件路径
            logger.info(f"> Segment {segment_name} - 包含文件数量: {len(segment)}")
            logger.info(f"> 包含的文件路径: {segment}")  # 查看具体是哪些文件（音频、标签、IMU等）
            
            
            logger.info("> Processing segment: %s", segment_name)
            
            # 初始化极端值检测器
            detector = AudioExtremeDetector()

            # Read audio file.
            audio_signal, sf = librosa.load(segment[0])
            # 检测原始音频极端值（振幅超出[-1,1]）- 未划分窗口前的统计
            orig_min = np.min(audio_signal)
            orig_max = np.max(audio_signal)
            orig_mean = np.mean(audio_signal)
            orig_std = np.std(audio_signal)
            extreme_mask = (audio_signal < -1.0) | (audio_signal > 1.0)
            extreme_count = np.sum(extreme_mask)
            
            # 记录原始音频数据统计特征（未划分窗口前）
            logger.info(
                f"【未划分窗口】原始音频{segment[0]}统计: "
                f"min={orig_min:.4f}, max={orig_max:.4f}, "
                f"mean={orig_mean:.4f}, std={orig_std:.4f}, "
                f"极端值数量={extreme_count}, 占比{extreme_count/len(audio_signal):.2%}"
            )
            
            if extreme_count > 0:
                logger.warning(
                    f"原始音频{segment[0]}存在极端值（超出[-1,1]）"
                )
            
            # 工具类检测
            audio_signal = detector.check_extreme(audio_signal, "原始音频加载")

            # 重采样
            audio_signal = librosa.resample(
                y=audio_signal,
                orig_sr=sf,
                target_sr=audio_sampling_frequency
            )
            
            # 记录重采样后的数据特征（未划分窗口前）
            resample_min = np.min(audio_signal)
            resample_max = np.max(audio_signal)
            resample_mean = np.mean(audio_signal)
            resample_std = np.std(audio_signal)
            resample_extreme_mask = (audio_signal < -1.0) | (audio_signal > 1.0)
            resample_extreme_count = np.sum(resample_extreme_mask)
            
            logger.info(
                f"【未划分窗口】重采样后音频统计: "
                f"min={resample_min:.4f}, max={resample_max:.4f}, "
                f"mean={resample_mean:.4f}, std={resample_std:.4f}, "
                f"极端值数量={resample_extreme_count}, 占比{resample_extreme_count/len(audio_signal):.2%}"
            )
            
            if resample_extreme_count > 0:
                logger.warning(
                    f"重采样后音频存在极端值（超出[-1,1]）"
                )
            
            # 工具类检测
            audio_signal = detector.check_extreme(audio_signal, "重采样后")

            # 标准化处理
            pre_norm_min = np.min(audio_signal)
            pre_norm_max = np.max(audio_signal)
            pre_norm_mean = np.mean(audio_signal)
            pre_norm_std = np.std(audio_signal)
            
            logger.info(
                f"【未划分窗口】标准化前音频统计: "
                f"min={pre_norm_min:.4f}, max={pre_norm_max:.4f}, "
                f"mean={pre_norm_mean:.4f}, std={pre_norm_std:.4f}"
            )
            
            # 执行标准化
            if pre_norm_std > 0:
                audio_signal = (audio_signal - pre_norm_mean) / pre_norm_std
            else:
                logger.warning("标准化时标准差为0，无法标准化，将使用原始信号")
            
            # 记录标准化后的数据特征（未划分窗口前）
            post_norm_min = np.min(audio_signal)
            post_norm_max = np.max(audio_signal)
            post_norm_mean = np.mean(audio_signal)
            post_norm_std = np.std(audio_signal)
            
            logger.info(
                f"【未划分窗口】标准化后音频统计: "
                f"min={post_norm_min:.4f}, max={post_norm_max:.4f}, "
                f"mean={post_norm_mean:.4f}, std={post_norm_std:.4f}"
            )
            
            # 工具类检测标准化后的数据
            audio_signal = detector.check_extreme(audio_signal, "标准化后")

            dataset_has_movement_data = len(segment) > 2

            # Read IMU files.
            imu_data = []

            if dataset_has_movement_data:
                for i in range(1, 10):
                    signal_axis_values = pd.read_csv(segment[i],
                                                    names=['axis_value']).axis_value.values
                    
                    # 检查IMU原始数据中的NaN和极端值（未划分窗口前）
                    if np.isnan(signal_axis_values).any():
                        nan_count = np.sum(np.isnan(signal_axis_values))
                        logger.warning(f"IMU文件{segment[i]}存在{nan_count}个NaN值，占比{nan_count/len(signal_axis_values):.2%}")
                        signal_axis_values = np.nan_to_num(signal_axis_values, nan=0.0)
                    
                    # 检查并记录IMU数据统计特征（未划分窗口前）
                    axis_min = np.min(signal_axis_values)
                    axis_max = np.max(signal_axis_values)
                    axis_mean = np.mean(signal_axis_values)
                    axis_std = np.std(signal_axis_values)
                    
                    logger.info(
                        f"【未划分窗口】IMU文件{segment[i]}统计: "
                        f"min={axis_min:.4f}, max={axis_max:.4f}, "
                        f"mean={axis_mean:.4f}, std={axis_std:.4f}"
                    )
                    
                    if axis_min < -1e6 or axis_max > 1e6:  # 假设IMU数据不会超过这个范围
                        logger.warning(f"IMU文件{segment[i]}存在极端值")
                    
                    data_sampling_frequency = available_datasets[dataset].imu_sampling_frequency
                    if data_sampling_frequency != movement_sampling_frequency:
                        sampling_relation = data_sampling_frequency / movement_sampling_frequency

                        signal_decimated = signal.decimate(signal_axis_values,
                                                        int(sampling_relation))
                        
                        # 记录降采样后的IMU数据统计（未划分窗口前）
                        decim_min = np.min(signal_decimated)
                        decim_max = np.max(signal_decimated)
                        decim_mean = np.mean(signal_decimated)
                        decim_std = np.std(signal_decimated)
                        logger.info(
                            f"【未划分窗口】IMU通道{i}降采样后统计: "
                            f"min={decim_min:.4f}, max={decim_max:.4f}, "
                            f"mean={decim_mean:.4f}, std={decim_std:.4f}"
                        )
                        
                        imu_data.append(signal_decimated)
                    else:
                        imu_data.append(signal_axis_values)

                if include_movement_magnitudes:
                    # 计算模值前再次检查是否有NaN
                    for i in range(9):
                        if np.isnan(imu_data[i]).any():
                            logger.warning(f"计算模值前，IMU通道{i}存在NaN值")
                            imu_data[i] = np.nan_to_num(imu_data[i], nan=0.0)
                            
                    accelerometer_magnitude = \
                        np.sqrt(imu_data[0] **2 + imu_data[1]** 2 + imu_data[2] **2)
                    gyroscope_magnitude = \
                        np.sqrt(imu_data[3]** 2 + imu_data[4] **2 + imu_data[5]** 2)
                    magnetometer_magnitude = \
                        np.sqrt(imu_data[6] **2 + imu_data[7]** 2 + imu_data[8] **2)
                    
                    # 记录模值数据统计（未划分窗口前）
                    logger.info(
                        f"【未划分窗口】加速度计模值统计: "
                        f"min={np.min(accelerometer_magnitude):.4f}, max={np.max(accelerometer_magnitude):.4f}, "
                        f"mean={np.mean(accelerometer_magnitude):.4f}, std={np.std(accelerometer_magnitude):.4f}"
                    )
                    logger.info(
                        f"【未划分窗口】陀螺仪模值统计: "
                        f"min={np.min(gyroscope_magnitude):.4f}, max={np.max(gyroscope_magnitude):.4f}, "
                        f"mean={np.mean(gyroscope_magnitude):.4f}, std={np.std(gyroscope_magnitude):.4f}"
                    )
                    logger.info(
                        f"【未划分窗口】磁力计模值统计: "
                        f"min={np.min(magnetometer_magnitude):.4f}, max={np.max(magnetometer_magnitude):.4f}, "
                        f"mean={np.mean(magnetometer_magnitude):.4f}, std={np.std(magnetometer_magnitude):.4f}"
                    )
                    
                    imu_data.append(accelerometer_magnitude)
                    imu_data.append(gyroscope_magnitude)
                    imu_data.append(magnetometer_magnitude)

            # 应用滤波器
            if filters:
                for filter in filters:
                    for channel in filter[1]:
                        filter_method = filter[0]
                        if filter[2] and dataset_has_movement_data:
                            # 滤波前检查IMU数据
                            if np.isnan(imu_data[channel]).any():
                                logger.warning(f"滤波前，IMU通道{channel}存在NaN值")
                                imu_data[channel] = np.nan_to_num(imu_data[channel], nan=0.0)
                            
                            # 记录滤波前IMU通道统计（未划分窗口前）
                            pre_filter_min = np.min(imu_data[channel])
                            pre_filter_max = np.max(imu_data[channel])
                            pre_filter_mean = np.mean(imu_data[channel])
                            pre_filter_std = np.std(imu_data[channel])
                            logger.info(
                                f"【未划分窗口】IMU通道{channel}滤波前统计: "
                                f"min={pre_filter_min:.4f}, max={pre_filter_max:.4f}, "
                                f"mean={pre_filter_mean:.4f}, std={pre_filter_std:.4f}"
                            )
                            
                            imu_data[channel] = filter_method(imu_data[channel])
                            
                            # 记录滤波后IMU通道统计（未划分窗口前）
                            post_filter_min = np.min(imu_data[channel])
                            post_filter_max = np.max(imu_data[channel])
                            post_filter_mean = np.mean(imu_data[channel])
                            post_filter_std = np.std(imu_data[channel])
                            logger.info(
                                f"【未划分窗口】IMU通道{channel}滤波后统计: "
                                f"min={post_filter_min:.4f}, max={post_filter_max:.4f}, "
                                f"mean={post_filter_mean:.4f}, std={post_filter_std:.4f}"
                            )
                            
                            # 滤波后检查IMU数据
                            if np.isnan(imu_data[channel]).any():
                                nan_count = np.sum(np.isnan(imu_data[channel]))
                                logger.warning(f"滤波后，IMU通道{channel}存在{nan_count}个NaN值")
                                imu_data[channel] = np.nan_to_num(imu_data[channel], nan=0.0)
                        else:
                            # 记录滤波前音频统计（未划分窗口前）
                            pre_filter_min = np.min(audio_signal)
                            pre_filter_max = np.max(audio_signal)
                            pre_filter_mean = np.mean(audio_signal)
                            pre_filter_std = np.std(audio_signal)
                            logger.info(
                                f"【未划分窗口】音频滤波前统计: "
                                f"min={pre_filter_min:.4f}, max={pre_filter_max:.4f}, "
                                f"mean={pre_filter_mean:.4f}, std={pre_filter_std:.4f}"
                            )
                            
                            # 对音频信号滤波
                            audio_signal = filter_method(audio_signal)
                            
                            # 记录滤波后音频统计（未划分窗口前）
                            post_filter_min = np.min(audio_signal)
                            post_filter_max = np.max(audio_signal)
                            post_filter_mean = np.mean(audio_signal)
                            post_filter_std = np.std(audio_signal)
                            logger.info(
                                f"【未划分窗口】音频滤波后统计: "
                                f"min={post_filter_min:.4f}, max={post_filter_max:.4f}, "
                                f"mean={post_filter_mean:.4f}, std={post_filter_std:.4f}"
                            )
                            
                            # 检测滤波后的极端值
                            filter_extreme_mask = (audio_signal < -1.0) | (audio_signal > 1.0)
                            filter_extreme_count = np.sum(filter_extreme_mask)
                            if filter_extreme_count > 0:
                                logger.warning(
                                    f"滤波后音频存在{filter_extreme_count}个极端值，"
                                    f"占比{filter_extreme_count/len(audio_signal):.2%}"
                                )
                            # 工具类检测
                            audio_signal = detector.check_extreme(audio_signal, f"滤波处理-{filter_method.__name__}")

            # Read labels file.
            df_segment_labels = pd.read_csv(
                segment[-1],
                sep='\t',
                names=["start", "end", "jm_event"])

            # Get windows from signals.
            audio_windows = get_windows_from_audio_signal(
                audio_signal,
                sampling_frequency=audio_sampling_frequency,
                window_width=window_width,
                window_overlap=window_overlap)
            
            # 【划分窗口后】音频整体统计
            if len(audio_windows) > 0:
                all_windows = np.concatenate(audio_windows)
                windowed_min = np.min(all_windows)
                windowed_max = np.max(all_windows)
                windowed_mean = np.mean(all_windows)
                windowed_std = np.std(all_windows)
                windowed_extreme_mask = (all_windows < -1.0) | (all_windows > 1.0)
                windowed_extreme_count = np.sum(windowed_extreme_mask)
                
                logger.info(
                    f"【划分窗口后】音频整体统计: "
                    f"min={windowed_min:.4f}, max={windowed_max:.4f}, "
                    f"mean={windowed_mean:.4f}, std={windowed_std:.4f}, "
                    f"极端值数量={windowed_extreme_count}, 占比{windowed_extreme_count/len(all_windows):.2%}"
                )
            
            # 添加日志：打印音频长度和窗口数量
            logger.info(f"Segment {segment_name} audio length: {len(audio_signal)/audio_sampling_frequency:.2f}s")
            logger.info(f"Generated {len(audio_windows)} windows for {segment_name}")
            
            
            # 检查音频窗口
            audio_has_nan = any(np.isnan(window).any() for window in audio_windows)
            logger.info(f"Segment {segment_name} audio windows have NaN: {audio_has_nan}")
            # 处理音频窗口中的NaN
            if audio_has_nan:
                logger.warning(f"处理音频窗口中的NaN值")
                audio_windows = np.nan_to_num(audio_windows, nan=0.0, posinf=0.0, neginf=0.0)
            
            
            imu_windows = []
            if dataset_has_movement_data:
                imu_windows = get_windows_from_imu_signals(
                    imu_data,
                    sampling_frequency=movement_sampling_frequency,
                    window_width=window_width,
                    window_overlap=window_overlap)

                # 【划分窗口后】IMU整体统计
                if len(imu_windows) > 0:
                    # 拼接所有IMU窗口数据
                    all_imu_windows = np.concatenate([np.concatenate(win) for win in imu_windows])
                    imu_windowed_min = np.min(all_imu_windows)
                    imu_windowed_max = np.max(all_imu_windows)
                    imu_windowed_mean = np.mean(all_imu_windows)
                    imu_windowed_std = np.std(all_imu_windows)
                    imu_extreme_mask = (all_imu_windows < -1e6) | (all_imu_windows > 1e6)
                    imu_extreme_count = np.sum(imu_extreme_mask)
                    
                    logger.info(
                        f"【划分窗口后】IMU整体统计: "
                        f"min={imu_windowed_min:.4f}, max={imu_windowed_max:.4f}, "
                        f"mean={imu_windowed_mean:.4f}, std={imu_windowed_std:.4f}, "
                        f"极端值数量={imu_extreme_count}, 占比{imu_extreme_count/len(all_imu_windows):.2%}"
                    )

                # 检查IMU窗口是否有NaN
                imu_has_nan = any(np.isnan(window).any() for imu_window in imu_windows for window in imu_window)
                logger.info(f"Segment {segment_name} IMU windows have NaN: {imu_has_nan}")
                
                # 检查IMU窗口是否有极端值
                imu_has_extreme = False
                if len(imu_windows) > 0:
                    imu_has_extreme = (np.any(all_imu_windows < -1e6) or np.any(all_imu_windows > 1e6))
                    if imu_has_extreme:
                        logger.warning(f"IMU窗口存在极端值")
                
                # 处理IMU窗口中的NaN和极端值
                if imu_has_nan or imu_has_extreme:
                    logger.warning(f"处理IMU窗口中的NaN和极端值")
                    imu_windows = np.nan_to_num(imu_windows, nan=0.0, posinf=1e6, neginf=-1e6)

                if len(audio_windows) - len(imu_windows) == 1:
                    logger.info('Removing last audio window in order to align with imu windows !')
                    audio_windows = audio_windows[:-1]

                assert len(audio_windows) == len(imu_windows),\
                    f'''Number of windows mismatched between audio
                        ({len(audio_windows)}) and IMU data ({len(imu_windows)}).'''

            # Get window labels.
            window_labels = get_windows_labels(
                df_segment_labels,
                len(audio_windows),
                window_width=window_width,
                window_overlap=window_overlap,
                label_overlapping_threshold=label_overlapping_threshold,
                no_event_class_name=no_event_class_name)

            segment_windows = []
            imu_channels = 0

            if dataset_has_movement_data:
                imu_channels = len(imu_windows[0])

            for i in range(len(audio_windows)):
                window_channels = []
                window_channels.append(audio_windows[i])

                if dataset_has_movement_data:
                    for c_i in range(imu_channels):
                        window_channels.append(imu_windows[i][c_i])
                segment_windows.append(window_channels)

            # Construct final results.
            X_dataset_segments[segment_name] = segment_windows
            y_dataset_segments[segment_name] = window_labels

        X[dataset] = X_dataset_segments
        y[dataset] = y_dataset_segments

    # Create cache item.
    cache.save(
        X,
        y,
        data_source_names=data_source_names,
        audio_sampling_frequency=audio_sampling_frequency,
        movement_sampling_frequency=movement_sampling_frequency,
        window_width=window_width,
        window_overlap=window_overlap,
        label_overlapping_threshold=label_overlapping_threshold,
        filter_noises=filter_noises,
        include_movement_magnitudes=include_movement_magnitudes,
        no_event_class_name=no_event_class_name,
        filters=filters)

    return X, y


def get_windows_from_audio_signal(
        signal,
        sampling_frequency,
        window_width,
        window_overlap):
    ''' Generate signal chunks using a fixed time window.

    Parameters
    ----------
    signal : NumPy array
        Signal values.
    sampling_frequency : int
        Number of samples per second.
    window_width : float
        Size of window in seconds used to split signals.
    window_overlap : float
        Overlapping proportion between to consecutive windows (0.00 - 1.00).

    Returns
    -------
    windows : list of lists.
        Extracted windows.
    '''
    # 确保输入信号中没有NaN
    if np.isnan(signal).any():
        logger.warning("音频信号中存在NaN值，已替换为0")
        signal = np.nan_to_num(signal, nan=0.0)
    
    # 记录窗口化前的信号统计（未划分窗口前）
    logger.info(
        f"【未划分窗口】音频窗口化前统计: "
        f"min={np.min(signal):.4f}, max={np.max(signal):.4f}, "
        f"mean={np.mean(signal):.4f}, std={np.std(signal):.4f}"
    )
        
    windows = librosa.util.frame(signal,
                                 frame_length=int(sampling_frequency * window_width),
                                 hop_length=int((1 - window_overlap) * int(sampling_frequency *
                                                                           window_width)),
                                 axis=0)

    return windows


def get_windows_from_imu_signals(
        imu_data,
        sampling_frequency,
        window_width,
        window_overlap):
    ''' Generate signal chunks using a fixed time window.

    Parameters
    ----------
    signal : NumPy array
        Signal values.
    sampling_frequency : int
        Number of samples per second.
    window_width : float
        Size of window in seconds used to split signals.
    window_overlap : float
        Overlapping proportion between to consecutive windows (0.00 - 1.00).

    Returns
    -------
    windows : list of lists.
        Extracted windows.
    '''

    hop_length = int((1 - window_overlap) * int(sampling_frequency * window_width))
    frame_length = int(sampling_frequency * window_width)

    signals = []
    for ix in range(len(imu_data)):
        # 确保每个IMU通道没有NaN
        if np.isnan(imu_data[ix]).any():
            logger.warning(f"IMU通道{ix}中存在NaN值，已替换为0")
            imu_data[ix] = np.nan_to_num(imu_data[ix], nan=0.0)
        
        # 记录IMU窗口化前的统计（未划分窗口前）
        logger.info(
            f"【未划分窗口】IMU通道{ix}窗口化前统计: "
            f"min={np.min(imu_data[ix]):.4f}, max={np.max(imu_data[ix]):.4f}, "
            f"mean={np.mean(imu_data[ix]):.4f}, std={np.std(imu_data[ix]):.4f}"
        )
            
        signals.append(
            librosa.util.frame(imu_data[ix],
                               frame_length=frame_length,
                               hop_length=hop_length,
                               axis=0))

    return list(map(list, zip(*signals)))


def get_windows_labels(
        labels,
        n_windows,
        window_width,
        window_overlap,
        label_overlapping_threshold,
        no_event_class_name):
    ''' Extract labels for each window.

    Parameters
    ----------
    labels : pandas DataFrame instance.
        Labels information including start, end and event.
    n_windows : int
        Number of windows.
    window_width : float
        Size of window in seconds used to split signals.
    window_overlap : float
        Percentage of overlapping between to consecutive windows (0-100%).
    label_overlapping_threshold : float
        Minimun threshold to assign a label to frame w.r.t. window width (0-100%).
    no_event_class_name : str
        Class name to represent the absense of an event of interest.

    Returns
    -------
    window_labels : list
        Corresponding label for each window.
    '''
    window_start = 0
    window_end = window_width

    window_labels = []

    labels['not_used'] = True

    # 记录标签分布情况
    label_distribution = labels['jm_event'].value_counts().to_dict()
    logger.info(f"原始标签分布: {label_distribution}")

    for i in range(n_windows):
        labels_matched = labels[(labels.start <= window_end) & (labels.end >= window_start)]

        if len(labels_matched) > 0:
            overlappings = []
            for index, label in labels_matched.iterrows():
                event_duration = label.end - label.start
                overlap_in_seconds = min(label.end, window_end) - max(label.start, window_start)
                overlappings.append((overlap_in_seconds,
                                     label.jm_event,
                                     index,
                                     event_duration))

            # Sort all labels with overlap.
            overlappings.sort(key=lambda tup: tup[0], reverse=True)

            exist_overlap_for_window = False
            for ix_o, overlap in enumerate(overlappings):
                # If the window contains the entire event, asign the label.
                window_contains_the_event = (overlap[0] / overlap[3]) == 1

                # If overlap % compared to window width reachs the threshold, asign the label.
                relative_overlap = (overlap[0] / window_width)
                overlap_reachs_threshold = relative_overlap >= label_overlapping_threshold

                # If any of created conditions is True, then asign the label to the window.
                if (window_contains_the_event or overlap_reachs_threshold):
                    exist_overlap_for_window = True

                    # If overlap is enough, the label of event with more overlap will be used.
                    window_labels.append(overlap[1])

                    # All events with enough overlap are used.
                    labels.loc[overlap[2], 'not_used'] = False

                    break

            if not exist_overlap_for_window:
                window_labels.append(no_event_class_name)
        else:
            window_labels.append(no_event_class_name)

        window_start = window_start + window_width * (1 - window_overlap)
        window_end = window_start + window_width

    # 记录窗口标签分布
    window_label_counts = pd.Series(window_labels).value_counts().to_dict()
    logger.info(f"窗口标签分布: {window_label_counts}")

    # not_used_labels = labels[(labels.jm_event != 'u') & (labels.not_used)]
    not_used_labels = labels[labels.not_used]
    if len(not_used_labels) > 0:
        logger.info('Some labels have not been used: %s', str(len(not_used_labels)))
        unused_distribution = not_used_labels['jm_event'].value_counts().to_dict()
        logger.info(f"未使用的标签分布: {unused_distribution}")
    else:
        logger.info('All labels have been used.')

    return window_labels