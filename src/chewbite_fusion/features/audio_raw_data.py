import numpy as np
from chewbite_fusion.features.base import BaseFeature

class AudioRawData(BaseFeature):
    def transform(self, X, y=None):
        raw_audio = []
        for file in X:  # 遍历每个文件
            # 对每个窗口的音频数据（window[0]）转为数组并添加通道维度（保持窗口列表结构）
            processed_windows = [np.expand_dims(np.array(window[0]), axis=-1) for window in file]
            raw_audio.append(processed_windows)  # 每个文件对应一个窗口列表
        return raw_audio