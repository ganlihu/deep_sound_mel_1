import numpy as np
import librosa  # 需要 librosa 库用于计算梅尔频谱
from chewbite_fusion.features.base import BaseFeature

class AudioMelSpectrogram(BaseFeature):
    def __init__(self, audio_sampling_frequency, movement_sampling_frequency):
        super().__init__(audio_sampling_frequency, movement_sampling_frequency)
        # 梅尔频谱参数（可根据需求调整）
        self.n_mels = 128  # 梅尔频段数量
        self.hop_length = 512  # 步长（控制时间帧数量）
        self.n_fft = 1024  # FFT窗口大小

    def transform(self, X, y=None):
        """
        输入：X 是文件列表，每个文件是窗口列表，每个窗口的音频数据在 window[0]
        输出：每个窗口的梅尔频谱，形状为 (时间帧, n_mels, 1)，保留窗口列表结构
        """
        mel_spectrograms = []
        for file in X:  # 遍历每个文件
            processed_windows = []
            for window in file:  # 遍历文件中的每个窗口
                audio_data = np.array(window[0])  # 窗口内的原始音频数据（1D数组）
                
                # 计算梅尔频谱（ librosa 会自动处理单通道音频）
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_data,
                    sr=self.audio_sampling_frequency,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
                
                # 转换为分贝值（更符合听觉特性）
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # 调整形状：(n_mels, 时间帧) → (时间帧, n_mels)，并添加通道维度 (1)
                mel_spec_db = mel_spec_db.T  # 转置后时间帧在前
                mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)  # 最后添加通道维度
                
                processed_windows.append(mel_spec_db)
            mel_spectrograms.append(processed_windows)
        return mel_spectrograms