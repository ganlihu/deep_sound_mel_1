import logging
import tensorflow as tf
from chewbite_fusion.models.deep_sound_mel import DeepSound
from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
# 1. 导入新的特征工厂（替换原来的 FeatureFactory_RawAudioData）
from chewbite_fusion.features.feature_factories import (
    FeatureFactory_MelSpectrogram,  # 仅梅尔频谱
    FeatureFactory_AudioMelFusion   # 原始音频 + 梅尔频谱
)

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    # 2. 根据特征类型调整模型输入尺寸：
    # - 仅梅尔频谱：输入形状为 (时间帧, 128, 1)，需模型支持2D输入
    # - 融合特征：需模型支持多输入（原始音频 + 梅尔频谱）
    # 这里以融合模型为例，假设 DeepSound 已修改为支持双输入
    return DeepSound(
        input_size=1800,  # 原始音频输入长度（保持不变）
        mel_input_shape=(None, 128),  # 梅尔频谱输入形状
        output_size=5,
        n_epochs=1,
        batch_size=5,
        set_sample_weights=True,
        feature_scaling=True
    )


@experiment()
def deep_sound_mel():
    """ Experiment with Deep Sound architecture (梅尔频谱/融合特征版本) """
    window_width = 0.3
    window_overlap = 0.5
    # 3. 生成数据集（音频采样率需与梅尔频谱计算一致）
    X, y = main(
        window_width=window_width,
        window_overlap=window_overlap,
        include_movement_magnitudes=False,
        audio_sampling_frequency=6000  # 保持与原设置一致
    )

    logger.info("生成的片段数量: %s", len(X['zavalla2022'].keys()))
    logger.info("片段编号示例: %s", list(X['zavalla2022'].keys())[:5])
    
    # 4. 初始化实验，指定新的特征工厂
    e = Experiment(
        get_model_instance,
        # 可选：FeatureFactory_MelSpectrogram（仅梅尔频谱）
        FeatureFactory_AudioMelFusion,  # 使用融合特征工厂
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deep_sound_mel_fusion',  # 重命名实验以区分
        manage_sequences=True,
        use_raw_data=True  # 关键：保留原始特征形状（不拼接）
    )
    e.run()