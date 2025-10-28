import os
import sys
import logging
import pickle
import gc  # 垃圾回收模块
from glob import glob
from datetime import datetime as dt
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
import sed_eval
import dcase_util
import tensorflow as tf  # 导入TensorFlow

from chewbite_fusion.data.utils import windows2events
from chewbite_fusion.experiments import settings
from chewbite_fusion.experiments.utils import set_random_init


# 自定义日志处理器，同时处理stdout和stderr
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


logger = logging.getLogger('yaer')


class Experiment:
    ''' Base class to represent an experiment using audio and movement signals. '''
    def __init__(self,
                 model_factory,
                 features_factory,
                 X,
                 y,
                 window_width,
                 window_overlap,
                 name,
                 audio_sampling_frequency=8000,
                 movement_sampling_frequency=100,
                 no_event_class='no-event',
                 manage_sequences=False,
                 model_parameters_grid={},
                 use_raw_data=True,
                 quantization=None,
                 data_augmentation=False):
        self.timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        self.model_factory = model_factory
        self.features_factory = features_factory
        self.X = X
        self.y = y
        self.window_width = window_width
        self.window_overlap = window_overlap
        self.name = name
        self.audio_sampling_frequency = audio_sampling_frequency
        self.movement_sampling_frequency = movement_sampling_frequency
        self.no_event_class = no_event_class
        self.manage_sequences = manage_sequences
        self.model_parameters_grid = model_parameters_grid
        self.use_raw_data = use_raw_data
        self.train_validation_segments = []
        self.quantization = quantization
        self.data_augmentation = data_augmentation
        self.max_window_length = 0  # 新增：训练集最大窗口长度

        # 创建实验路径
        self.path = os.path.join(settings.experiments_path, name, self.timestamp)
        os.makedirs(self.path, exist_ok=True)
        self.models_dir = os.path.join(self.path, "trained_models")
        os.makedirs(self.models_dir, exist_ok=True)

        # 配置日志（同时输出到文件和控制台，捕获所有输出）
        logger.handlers = []  # 清空现有处理器
        logger.setLevel(logging.DEBUG)  # 设置最低日志级别为DEBUG

        # 文件处理器 - 保存所有日志到文件
        fileHandler = logging.FileHandler(f"{self.path}/experiment.log")
        # 控制台处理器 - 在控制台显示
        consoleHandler = logging.StreamHandler()

        # 统一日志格式
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)

        # 设置处理器日志级别
        fileHandler.setLevel(logging.DEBUG)
        consoleHandler.setLevel(logging.DEBUG)

        # 添加处理器
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)

        # 重定向stdout和stderr到日志
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

        # 设置随机种子
        set_random_init()

    def __del__(self):
        # 恢复stdout和stderr
        try:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
        except:
            pass

    def calculate_max_window_length(self):
        '''计算训练集中所有样本的最大窗口长度，用于统一对齐'''
        all_window_lengths = []
        # 遍历训练验证片段的真实标签窗口长度
        for key in self.X.keys():
            seg_id = int(key.split('_')[1])
            if seg_id in self.train_validation_segments:  # 仅考虑训练集样本
                if self.manage_sequences:
                    window_len = len(self.y[key])  # 每个样本的窗口数
                    all_window_lengths.append(window_len)
                    logger.debug(f"片段 {key} 窗口长度: {window_len}")
        
        if all_window_lengths:
            self.max_window_length = max(all_window_lengths)
            logger.info(f"训练集最大窗口长度确定为: {self.max_window_length}")
        else:
            logger.warning("未找到训练集样本，无法计算最大窗口长度")
            self.max_window_length = 0

    def run(self):
        ''' 运行实验并保存相关信息 '''
        try:
            self.X = self.X['zavalla2022']
            self.y = self.y['zavalla2022']

            # 验证片段ID解析
            for k in self.X.keys():
                try:
                    seg_id = int(k.split('_')[1])
                    logger.debug(f'Segment {k} parsed to ID {seg_id}')
                except:
                    logger.error(f'Failed to parse segment ID from {k}')

            # 折叠划分（基于反刍片段的分层随机抽样）
            folds = {
                '1': [40, 41, 7, 37, 26],
                '2': [24, 17, 48, 21, 5],
                '3': [33, 52, 23, 4, 15],
                '4': [36, 49, 18, 28, 50],
                '5': [35, 27, 44, 20, 9],
                '6': [51, 31, 3, 16, 42],
                '7': [22, 39, 32, 45, 34],
                '8': [2, 8, 30, 29, 1],
                '9': [19, 10, 6, 43, 47, 13],
                '10': [11, 14, 12, 38, 25, 46]
        }

            self.train_validation_segments = [seg for fold in folds.values() for seg in fold]
            # 计算训练集最大窗口长度（关键新增步骤）
            self.calculate_max_window_length()

            # 记录训练验证片段信息
            logger.info('train_validation_segments count: %d', len(self.train_validation_segments))
            logger.info('train_validation_segments: %s', self.train_validation_segments)

            hash_method_instance = hashlib.new('sha256')
            params_results = {}
            full_grid = list(ParameterGrid(self.model_parameters_grid))

            if len(full_grid) > 1:
                for params_combination in full_grid:
                    if params_combination:
                        logger.info('Running folds for parameters combination: %s.', params_combination)
                    else:
                        logger.info('Running folds without grid search.')

                    # 生成参数哈希值
                    hash_method_instance.update(str(params_combination).encode())
                    params_combination_hash = hash_method_instance.hexdigest()

                    # 执行k折交叉验证
                    params_combination_result = self.execute_kfoldcv(
                        folds=folds,
                        is_grid_search=True,
                        parameters_combination=params_combination
                    )

                    params_results[params_combination_hash] = (params_combination_result, params_combination)

                # 选择最佳参数组合
                best_params_combination = max(params_results.values(), key=lambda i: i[0])[1]
                logger.info('-' * 25)
                logger.info('>>> All params combination values: %s <<<', str(params_results))
                logger.info('-' * 25)
                logger.info('>>> Best params combination: %s <<<', best_params_combination)
            else:
                logger.info('-' * 25)
                logger.info('>>> Skipping grid search! No params dict provided. <<<')
                best_params_combination = full_grid[0] if full_grid else {}

            # 使用最佳参数执行最终交叉验证
            self.execute_kfoldcv(
                folds=folds,
                is_grid_search=False,
                parameters_combination=best_params_combination
            )
        finally:
            # 确保在实验结束时恢复标准输出
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

    def execute_kfoldcv(self, folds, is_grid_search, parameters_combination):
        ''' 使用特定参数执行k折交叉验证 '''
        fold_metrics = []
        for ix_fold, fold in folds.items():
            logger.info('Running fold number %s.', ix_fold)
            total_folds = len(folds)
            logger.info(f'当前训练折数：{ix_fold}/{total_folds}')
            # 划分训练/测试片段
            test_fold_keys = [k for k in self.X.keys() if int(k.split('_')[1]) in fold]
            train_fold_keys = [k for k in self.X.keys() if int(k.split('_')[1]) not in fold]
            train_fold_keys = [k for k in train_fold_keys if int(k.split('_')[1]) in self.train_validation_segments]

            # 记录训练片段信息
            logger.info('Train fold keys count: %d', len(train_fold_keys))
            logger.info('Train fold keys: %s.', str(train_fold_keys))

            # 准备训练数据
            X_train = []
            y_train = []
            for train_signal_key in train_fold_keys:
                if self.manage_sequences:
                    X_train.append(self.X[train_signal_key])
                    y_train.append(self.y[train_signal_key])
                else:
                    X_train.extend(self.X[train_signal_key])
                    y_train.extend(self.y[train_signal_key])

            # 检查特征与标签数量匹配
            for i, (x_seg, y_seg) in enumerate(zip(X_train, y_train)):
                if len(x_seg) != len(y_seg):
                    logger.error(f"片段 {train_fold_keys[i]} 特征与标签数量不匹配：{len(x_seg)} vs {len(y_seg)}")        

            # 记录窗口数量信息
            for i, key in enumerate(train_fold_keys):
                window_count = len(X_train[i]) if self.manage_sequences else 0
                logger.info(f"Segment {key} has {window_count} windows")
            logger.info(f"Total X_train length (segments): {len(X_train)}")
            logger.info(f"Total windows across all segments: {sum(len(seg) for seg in X_train)}")

            # 数据增强
            if self.data_augmentation:
                from augly.audio import functional
                # 计算类别分布
                all_y = []
                n_labels = 0
                for i_file in range(len(X_train)):
                    for i_window in range(len(X_train[i_file])):
                        if y_train[i_file][i_window] != 'no-event':
                            all_y.append(y_train[i_file][i_window])
                            n_labels += 1
                unique, counts = np.unique(all_y, return_counts=True)
                classes_probs = dict(zip(unique, counts / n_labels))

                # 复制原始数据
                import copy
                X_augmented = copy.deepcopy(X_train)
                y_augmented = copy.deepcopy(y_train)

                # 增强处理
                for i_file in range(len(X_train)):
                    during_event = False
                    discard_event = False
                    for i_window in range(len(X_train[i_file])):
                        window_label = y_train[i_file][i_window]

                        if window_label == 'no-event':
                            during_event = False
                            discard_event = False
                        elif not during_event and window_label not in ['no-event', 'bite', 'chew-bite']:
                            during_event = True
                            # 对多数类事件进行随机丢弃
                            if np.random.rand() <= classes_probs[window_label] * 2:
                                discard_event = True

                        if during_event and discard_event:
                            # 零值替换并标记为无事件
                            for i_channel in range(len(X_train[i_file][i_window])):
                                window_len = len(X_augmented[i_file][i_window][i_channel])
                                X_augmented[i_file][i_window][i_channel] = np.zeros(window_len)
                                y_augmented[i_file][i_window] = 'no-event'
                        else:
                            # 添加背景噪音
                            for i_channel in range(len(X_train[i_file][i_window])):
                                sample_rate = 6000 if i_channel == 0 else 100
                                window = X_augmented[i_file][i_window][i_channel]
                                X_augmented[i_file][i_window][i_channel] = \
                                    functional.add_background_noise(window, sample_rate, snr_level_db=20)[0]

                logger.info('Applying data augmentation !')
                logger.info(f"数据增强前训练样本数: {len(X_train)}")
                X_train.extend(X_augmented)
                y_train.extend(y_augmented)
                logger.info(f"数据增强后训练样本数: {len(X_train)}")

            # 标签编码
            self.target_encoder = LabelEncoder()
            unique_labels = np.unique(np.hstack(y_train))
            self.target_encoder.fit(unique_labels)

            # 记录标签映射关系
            logger.info("标签映射关系：")
            for idx, label in enumerate(self.target_encoder.classes_):
                logger.info(f"类别编号 {idx} → 原始标签 '{label}'")

            # 编码训练标签
            if self.manage_sequences:
                y_train_enc = [self.target_encoder.transform(file_labels) for file_labels in y_train]
            else:
                y_train_enc = self.target_encoder.transform(y_train)

            # 检查编码标签范围
            all_enc_labels = np.hstack(y_train_enc)
            max_label = np.max(all_enc_labels) if len(all_enc_labels) > 0 else -1
            if max_label >= len(self.target_encoder.classes_):
                logger.error(f"编码后的标签超出范围：最大值{max_label}，有效类别数{len(self.target_encoder.classes_)}")

            # 创建模型实例
            model_instance = self.model_factory(parameters_combination)
            self.model = model_instance
            self.set_model_output_path(ix_fold, is_grid_search)

            # 特征处理与模型训练
            funnel = Funnel(
                self.features_factory,
                model_instance,
                self.audio_sampling_frequency,
                self.movement_sampling_frequency,
                self.use_raw_data
            )

            logger.info(f"X_train 长度（样本数）：{len(X_train)}")
            logger.info(f"第一个样本形状：{len(X_train[0]) if X_train else None}")

            funnel.fit(X_train, y_train_enc)

            # 量化处理
            if self.quantization:
                for ix_layer, layer in enumerate(funnel.model.model.layers):
                    w = layer.get_weights()
                    w = [i.astype(self.quantization) for i in w]
                    funnel.model.model.layers[ix_layer].set_weights(w)
                logger.info(f'量化处理已应用，类型：{str(self.quantization)}')

            # 初始化当前折的预测结果字典
            current_fold_predictions = {}
            
            # 初始化列表用于收集当前折的真实标签和预测结果
            y_true_all = []
            y_pred_all = []

            # 模型预测
            logger.info(f"当前折测试样本总数: {len(test_fold_keys)}，编号: {test_fold_keys}")
            for test_signal_key in test_fold_keys:
                # 获取测试数据片段
                X_test_segment = [self.X[test_signal_key]] if self.manage_sequences else self.X[test_signal_key]
                y_test_segment = self.y[test_signal_key]

                # 模型预测
                y_pred_segment = funnel.predict(X_test_segment)

                # 输出预测结果的原始信息（包括维度）
                logger.info(f"预测结果原始形状: {np.shape(y_pred_segment)}, 维度: {y_pred_segment.ndim}")
                
                # 修复展平逻辑：根据预测结果的实际维度动态处理
                if self.manage_sequences:
                    # 情况1：如果是二维数组且第一维为1（如(1, 294)），则取第一个元素后展平
                    if y_pred_segment.ndim == 2 and y_pred_segment.shape[0] == 1:
                        y_pred_segment_flat = np.ravel(y_pred_segment[0])
                    # 情况2：如果是一维数组（如(294,)），直接展平
                    elif y_pred_segment.ndim == 1:
                        y_pred_segment_flat = np.ravel(y_pred_segment)
                    # 其他异常情况：强制展平并报警
                    else:
                        logger.warning(f"预测结果维度异常（{y_pred_segment.ndim}维），强制展平")
                        y_pred_segment_flat = np.ravel(y_pred_segment)
                else:
                    y_pred_segment_flat = y_pred_segment.flatten()

                # 输出展平后的形状和长度
                logger.info(f"展平后预测结果形状: {np.shape(y_pred_segment_flat)}, 长度: {len(y_pred_segment_flat)}")

                # 打印当前片段的真实标签和预测结果长度
                logger.info(f"片段 {test_signal_key}：真实标签数={len(y_test_segment)}, 预测结果数={len(y_pred_segment_flat)}")

                # 记录预测窗口数
                pred_window_count = len(y_pred_segment_flat)
                logger.info(f"测试样本 {test_signal_key} 预测窗口数: {pred_window_count}")
                
                # 收集到当前折的列表
                y_true_all.extend(y_test_segment)
                y_pred_all.extend(y_pred_segment_flat)

                # 逆编码预测结果
                y_signal_pred_labels = self.target_encoder.inverse_transform(y_pred_segment_flat)

                # 保存预测结果时记录真实标签窗口数
                true_window_count = len(y_test_segment)
                logger.info(f"测试样本 {test_signal_key} 真实标签窗口数: {true_window_count}")
                current_fold_predictions[test_signal_key] = [y_test_segment, y_signal_pred_labels]

            # 打印当前折的总真实标签和预测结果数量
            logger.info(f"折 {ix_fold} 总统计：真实标签总数={len(y_true_all)}, 预测结果总数={len(y_pred_all)}")

            # 每折结束后保存当前折的结果（使用修改后的save_predictions）
            self.save_predictions(current_fold_predictions, ix_fold)

            # 每折结束释放资源
            logger.info(f"折 {ix_fold} 训练/测试完成，开始释放资源...")
            # 删除模型和数据
            if hasattr(self, 'model'):
                del self.model
            if 'funnel' in locals():
                del funnel
            if 'model_instance' in locals():
                del model_instance
            if 'X_train' in locals():
                del X_train
            if 'y_train' in locals():
                del y_train
            if 'y_train_enc' in locals():
                del y_train_enc
            if 'X_augmented' in locals():
                del X_augmented
                del y_augmented
            # 清除Keras会话和GPU内存
            tf.keras.backend.clear_session()
            gc.collect()
            try:
                for gpu in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.reset_memory_stats(gpu.name)
                logger.info("GPU内存统计已重置")
            except Exception as e:
                logger.warning(f"重置GPU内存统计失败：{e}")
            logger.info(f"折 {ix_fold} 资源释放完成\n")

        # 评估结果
        logger.info('-' * 25)
        logger.info('Fold iterations finished !. Starting evaluation phase.')

        unique_labels = list(set(np.concatenate([self.y[k] for k in self.y.keys()])))
        if self.no_event_class in unique_labels:
            unique_labels.remove(self.no_event_class)

        if is_grid_search:
            fold_metrics_result = self.evaluate(unique_labels=unique_labels, folds=folds, verbose=False)
        else:
            logger.info('-' * 50)
            logger.info('***** Classification results *****')
            fold_metrics_result = self.evaluate(unique_labels=unique_labels, folds=folds, verbose=True)

        return fold_metrics_result

    def save_predictions(self, fold_labels_predictions, fold_ix):
        '''保存预测结果，将真实标签和预测结果统一调整到训练集最大窗口长度'''
        # 为每个折创建单独的目录
        fold_path = os.path.join(self.path, f'fold_{fold_ix}')
        os.makedirs(fold_path, exist_ok=True)
        
        # 验证当前折的样本数量
        expected = 12 if fold_ix == '5' else 10
        actual = len(fold_labels_predictions)
        logger.info(f"折 {fold_ix} 进入窗口转换的样本数: {actual}，预期: {expected}")
        if actual != expected:
            logger.warning(f"折 {fold_ix} 样本数异常：{actual}（预期{expected}）")

        df = pd.DataFrame(columns=['segment', 'y_true', 'y_pred'])
        for segment in fold_labels_predictions.keys():
            y_true = fold_labels_predictions[segment][0]
            y_pred = fold_labels_predictions[segment][1]
            target_length = self.max_window_length

            # 1. 若未计算出最大长度，则退化为截断到最短（避免报错）
            if target_length == 0:
                target_length = min(len(y_true), len(y_pred))
                logger.warning(f"未找到训练集最大窗口长度，使用当前样本最短长度: {target_length}")

            # 2. 调整真实标签长度到目标长度
            if len(y_true) > target_length:
                y_true = y_true[:target_length]  # 截断过长
                logger.info(f"片段 {segment} 真实标签过长，截断到 {target_length}（原长度{len(y_true)}）")
            elif len(y_true) < target_length:
                # 填充no-event到目标长度
                pad_length = target_length - len(y_true)
                y_true = np.pad(y_true, (0, pad_length), 
                                mode='constant', 
                                constant_values=self.no_event_class)
                logger.info(f"片段 {segment} 真实标签过短，填充到 {target_length}（原长度{len(y_true)-pad_length}）")

            # 3. 调整预测结果长度到目标长度（与真实标签保持一致）
            if len(y_pred) > target_length:
                y_pred = y_pred[:target_length]  # 截断过长
                logger.info(f"片段 {segment} 预测结果过长，截断到 {target_length}（原长度{len(y_pred)}）")
            elif len(y_pred) < target_length:
                # 填充no-event到目标长度
                pad_length = target_length - len(y_pred)
                y_pred = np.pad(y_pred, (0, pad_length), 
                                mode='constant', 
                                constant_values=self.no_event_class)
                logger.info(f"片段 {segment} 预测结果过短，填充到 {target_length}（原长度{len(y_pred)-pad_length}）")

            # 4. 验证长度一致性
            assert len(y_true) == len(y_pred) == target_length, \
                f"片段 {segment} 调整后长度不一致: 真实标签{len(y_true)} vs 预测结果{len(y_pred)}（目标{target_length}）"

            # 5. 保存到DataFrame
            _df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
            _df['segment'] = segment
            df = pd.concat([df, _df], ignore_index=True)

            # 6. 转换窗口为事件并保存
            logger.info(f"样本 {segment} - 开始真实标签窗口转事件")
            df_labels = windows2events(y_true, self.window_width, self.window_overlap)
            df_labels = df_labels[df_labels.label != self.no_event_class]
            df_labels.to_csv(os.path.join(fold_path, f'{segment}_true.txt'),
                             sep='\t', header=False, index=False)

            logger.info(f"样本 {segment} - 开始预测结果窗口转事件")
            df_predictions = windows2events(y_pred, self.window_width, self.window_overlap)
            df_predictions = df_predictions[df_predictions.label != self.no_event_class]
            df_predictions.to_csv(os.path.join(fold_path, f'{segment}_pred.txt'),
                                  sep='\t', header=False, index=False)

        df.to_csv(os.path.join(fold_path, 'fold_labels_and_predictions.csv'), index=False)
        logger.info(f"折 {fold_ix} 预测结果已保存到: {fold_path}")

    def evaluate(self, unique_labels, folds, verbose=True):
        ''' 评估模型性能 '''
        final_metric = 'f_measure'
        fold_metrics_detail = {}
        fold_metrics = []

        for ix_fold, fold in folds.items():
            # 从当前折的目录中获取文件
            fold_path = os.path.join(self.path, f'fold_{ix_fold}')
            target_files = glob(os.path.join(fold_path, 'recording_*_true.txt'))
            
            file_list = []
            fold_files = [f for f in target_files if int(os.path.basename(f).split('_')[1]) in fold]
            for file in fold_files:
                pred_file = file.replace('true', 'pred')
                file_list.append({'reference_file': file, 'estimated_file': pred_file})

            data = []
            all_data = dcase_util.containers.MetaDataContainer()
            for file_pair in file_list:
                reference_event_list = sed_eval.io.load_event_list(file_pair['reference_file'])
                estimated_event_list = sed_eval.io.load_event_list(file_pair['estimated_file'])
                data.append({'reference_event_list': reference_event_list,
                             'estimated_event_list': estimated_event_list})
                all_data += reference_event_list

            # 计算片段级和事件级指标
            segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
                event_label_list=unique_labels,
                time_resolution=settings.segment_width_value
            )
            event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
                event_label_list=unique_labels,
                t_collar=settings.collar_value
            )

            for file_pair in data:
                segment_based_metrics.evaluate(
                    reference_event_list=file_pair['reference_event_list'],
                    estimated_event_list=file_pair['estimated_event_list']
                )
                event_based_metrics.evaluate(
                    reference_event_list=file_pair['reference_event_list'],
                    estimated_event_list=file_pair['estimated_event_list']
                )

            # 保存指标到当前折的目录
            metrics = {
                'segment_based_metrics': segment_based_metrics,
                'event_based_metrics': event_based_metrics
            }
            with open(os.path.join(fold_path, f'experiment_metrics_fold_{ix_fold}.pkl'), 'wb') as handle:
                pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # 提取指标结果
            segment_metrics = segment_based_metrics.results_overall_metrics()
            event_metrics = event_based_metrics.results_overall_metrics()

            if verbose:
                logger.info('### Segment based metrics (fold %s) ###', ix_fold)
                logger.info(segment_based_metrics)
                logger.info('')
                logger.info('### Event based metrics (fold %s) ###', ix_fold)
                logger.info(event_based_metrics)
                logger.info('-' * 20)

            fold_metrics_detail[ix_fold] = {
                'event_score': event_metrics[final_metric],
                'segment_score': segment_metrics[final_metric]
            }
            fold_metrics.append(event_metrics[final_metric][final_metric])

        # 保存整体指标
        with open(os.path.join(self.path, 'experiment_overall_metrics.pkl'), 'wb') as handle:
            pickle.dump(fold_metrics_detail, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # 计算均值和标准差
        folds_mean = np.round(np.mean(fold_metrics), 6) if fold_metrics else 0.0
        folds_std = np.round(np.std(fold_metrics), 6) if fold_metrics else 0.0

        if verbose:
            logger.info('### Event based overall metrics ###')
            logger.info(f'F1 score (micro) mean for events: {folds_mean}')
            logger.info(f'F1 score (micro) standard deviation for events: {folds_std}')
            logger.info('-' * 20)

        return folds_mean

    def set_model_output_path(self, n_fold, is_grid_search=False):
        ''' 设置模型输出路径 '''
        output_logs_path = os.path.join(self.path, f'logs_fold_{n_fold}')
        output_model_checkpoint_path = os.path.join(self.path, f'model_checkpoints_fold_{n_fold}')

        # 非网格搜索时检查路径是否存在
        if not is_grid_search:
            if os.path.exists(output_logs_path):
                raise FileExistsError('Model output logs path already exists!')
            if os.path.exists(output_model_checkpoint_path):
                raise FileExistsError('Model output checkpoints path already exists!')

        # 创建路径并赋值给模型
        os.makedirs(output_logs_path, exist_ok=True)
        self.model.output_logs_path = output_logs_path

        os.makedirs(output_model_checkpoint_path, exist_ok=True)
        self.model.output_path_model_checkpoints = output_model_checkpoint_path


class Funnel:
    ''' 类似sklearn Pipeline的接口，用于并行特征处理 '''
    def __init__(self,
                 features_factory,
                 model_instance,
                 audio_sampling_frequency,
                 movement_sampling_frequency,
                 use_raw_data=False):
        self.features = features_factory(
            audio_sampling_frequency,
            movement_sampling_frequency
        ).features
        self.model = model_instance
        self.use_raw_data = use_raw_data

    def fit(self, X, y):
        ''' 拟合特征并训练模型 '''
        X_features = []
        for feature in self.features:
            logger.info(f'Processing the feature {feature.feature.__class__.__name__}.')
            X_features.append(feature.fit_transform(X, y))

        if not self.use_raw_data:
            X_features = np.concatenate(X_features, axis=1)

        logger.info('Training model ...')
        self.model.fit(X_features, y)

    def predict(self, X):
        ''' 特征转换并预测 '''
        X_features = []
        for feature in self.features:
            X_features.append(feature.transform(X))

        if not self.use_raw_data:
            X_features = np.concatenate(X_features, axis=1)

        return self.model.predict(X_features)