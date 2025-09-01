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
    return DeepSound(input_size=1800,
                     output_size=4,
                     n_epochs=1500,
                     batch_size=10,
                     training_reshape=True,
                     set_sample_weights=True,
                     feature_scaling=True)


@experiment()
def deep_sound():
    """ Experiment with Deep Sound architecture.
    """
    # 启用数值检查（新增此行）
    # tf.debugging.enable_check_numerics()  # 放在这里！
    
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000
                # 以下句子是添加的invalidate_cache=True
                # invalidate_cache=True
               )
    
    
    # 在此处添加日志，打印片段数量和示例编号
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