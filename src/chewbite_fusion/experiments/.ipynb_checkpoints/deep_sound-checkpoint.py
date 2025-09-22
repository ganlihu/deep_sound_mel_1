import os
# 设置可见GPU设备（需在导入tensorflow前执行）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import logging
import tensorflow as tf  # 新增：导入TensorFlow
from chewbite_fusion.models.deep_sound import DeepSound
from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features.feature_factories import FeatureFactory_RawAudioData

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    # 核心修改：适配多GPU，按GPU数量调整批次大小（3个GPU × 单卡基础批次5）
    return DeepSound(input_size=1800,
                     output_size=5,
                     n_epochs=1,
                     batch_size=9,  # 多GPU批次调整：3×5=15
                     set_sample_weights=True,
                     feature_scaling=True,
                     gpus=[0, 1, 2]  # 传递可用GPU列表
                     )


@experiment()
def deep_sound():
    """ Experiment with Deep Sound architecture.
    """
    # 启用数值检查（如需调试数值问题可取消注释）
    # tf.debugging.enable_check_numerics()
    
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000
                # 如需重新生成数据集可取消注释
                # invalidate_cache=True
               )
    
    
    # 日志：打印片段数量和示例编号
    logger.info("生成的片段数量: %s", len(X['zavalla2022'].keys()))
    logger.info("片段编号示例: %s", list(X['zavalla2022'].keys())[:5])
    e = Experiment(get_model_instance,
                   FeatureFactory_RawAudioData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_sound',
                   manage_sequences=True,
                   use_raw_data=True)
    e.run()