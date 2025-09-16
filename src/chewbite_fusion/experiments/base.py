import os
import logging
import pickle
import gc
from glob import glob
from datetime import datetime as dt
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import ParameterGrid
import sed_eval
import dcase_util
import tensorflow as tf

from chewbite_fusion.data.utils import windows2events
from chewbite_fusion.experiments import settings
from chewbite_fusion.experiments.utils import set_random_init


logger = logging.getLogger('yaer')


class Experiment:
    '''Base class to represent an experiment using audio and movement signals.'''
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
        self.callbacks = []  # 新增：存储回调函数

        self.path = os.path.join(settings.experiments_path, name, self.timestamp)
        os.makedirs(self.path, exist_ok=True)

        # 配置日志（文件和标准输出）
        logger.handlers = []
        fileHandler = logging.FileHandler(f"{self.path}/experiment.log")
        consoleHandler = logging.StreamHandler()  # 新增控制台输出
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
        logger.setLevel(logging.INFO)

        set_random_init()

    # 新增：添加回调函数的方法
    def add_callbacks(self, callbacks):
        self.callbacks.extend(callbacks)

    def run(self):
        '''运行实验并保存相关信息'''
        self.X = self.X['zavalla2022']
        self.y = self.y['zavalla2022']

        for k in self.X.keys():
            try:
                seg_id = int(k.split('_')[1])
                logger.debug(f'Segment {k} parsed to ID {seg_id}')
            except:
                logger.error(f'Failed to parse segment ID from {k}')

        folds = {
            '1': [40, 41, 7, 37, 26, 24, 17, 48, 21, 5],
            '2': [33, 52, 23, 4, 15, 36, 49, 18, 28, 50],
            '3': [35, 27, 44, 20, 9, 51, 31, 3, 16, 42],
            '4': [22, 39, 32, 45, 34, 2, 8, 30, 29, 1],
            '5': [19, 10, 6, 43, 47, 13, 11, 14, 12, 38, 25, 46]
        }

        self.train_validation_segments = [seg for fold in folds.values() for seg in fold]
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

                hash_method_instance.update(str(params_combination).encode())
                params_combination_hash = hash_method_instance.hexdigest()

                params_combination_result = self.execute_kfoldcv(
                    folds=folds,
                    is_grid_search=True,
                    parameters_combination=params_combination
                )

                params_results[params_combination_hash] = (params_combination_result, params_combination)

            best_params_combination = max(params_results.values(), key=lambda i: i[0])[1]
            logger.info('-' * 25)
            logger.info('>>> All params combination values: %s <<<', str(params_results))
            logger.info('-' * 25)
            logger.info('>>> Best params combination: %s <<<', best_params_combination)
        else:
            logger.info('-' * 25)
            logger.info('>>> Skipping grid search! No params dict provided. <<<')
            best_params_combination = full_grid[0] if full_grid else {}

        self.execute_kfoldcv(
            folds=folds,
            is_grid_search=False,
            parameters_combination=best_params_combination
        )

    def execute_kfoldcv(self, folds, is_grid_search, parameters_combination):
        '''使用特定参数执行k折交叉验证，新增窗口级评估'''
        signal_predictions = {}
        window_metrics = []  # 存储各折的窗口级准确率

        for ix_fold, fold in folds.items():
            logger.info('Running fold number %s.', ix_fold)

            # 划分训练/测试片段
            test_fold_keys = [k for k in self.X.keys() if int(k.split('_')[1]) in fold]
            train_fold_keys = [k for k in self.X.keys() if int(k.split('_')[1]) not in fold]
            train_fold_keys = [k for k in train_fold_keys if int(k.split('_')[1]) in self.train_validation_segments]

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
                all_y = []
                n_labels = 0
                for i_file in range(len(X_train)):
                    for i_window in range(len(X_train[i_file])):
                        if y_train[i_file][i_window] != 'no-event':
                            all_y.append(y_train[i_file][i_window])
                            n_labels += 1
                unique, counts = np.unique(all_y, return_counts=True)
                classes_probs = dict(zip(unique, counts / n_labels))

                import copy
                X_augmented = copy.deepcopy(X_train)
                y_augmented = copy.deepcopy(y_train)

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
                            if np.random.rand() <= classes_probs[window_label] * 2:
                                discard_event = True

                        if during_event and discard_event:
                            for i_channel in range(len(X_train[i_file][i_window])):
                                window_len = len(X_augmented[i_file][i_window][i_channel])
                                X_augmented[i_file][i_window][i_channel] = np.zeros(window_len)
                                y_augmented[i_file][i_window] = 'no-event'
                        else:
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

                # 数据增强后标签检查
                all_augmented_labels = np.hstack(y_train) if isinstance(y_train[0], (list, np.ndarray)) else y_train
                unique_augmented = np.unique(all_augmented_labels)
                logger.info(f"===== 数据增强后标签检查 =====")
                logger.info(f"增强后去重标签: {unique_augmented}")
                logger.info(f"增强后标签数量: {len(unique_augmented)}")
                logger.info(f"=========================")

            # 标签编码
            self.target_encoder = LabelEncoder()
            # 展平y_train获取所有原始标签（处理序列数据）
            all_raw_labels = np.hstack(y_train) if isinstance(y_train[0], (list, np.ndarray)) else y_train
            unique_labels = np.unique(all_raw_labels)

            # 检查原始标签中是否存在无效值
            logger.info(f"所有原始标签值: {unique_labels}")
            
            # 新增：训练数据标签数量检查
            logger.info(f"===== 标签数量核心检查 =====")
            logger.info(f"训练数据中所有原始标签（前100个）: {all_raw_labels[:100]}")  # 打印部分标签
            logger.info(f"去重后的标签列表: {unique_labels}")
            logger.info(f"去重后的标签数量: {len(unique_labels)}")  # 关键：检查是否为5
            logger.info(f"=========================")

            self.target_encoder.fit(unique_labels)

            # 打印原始标签与编码映射
            logger.info(f"折 {ix_fold} - 原始标签（去重）: {unique_labels}")
            logger.info(f"折 {ix_fold} - 标签编码映射:")
            for raw_label, encoded in zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)):
                logger.info(f"  原始标签 '{raw_label}' → 编码值 {encoded}")
            logger.info(f"折 {ix_fold} - 实际类别数: {len(unique_labels)}, 编码最大值: {max(self.target_encoder.transform(self.target_encoder.classes_)) if len(unique_labels) > 0 else '无'}")

            # 编码训练标签
            if self.manage_sequences:
                y_train_enc = [self.target_encoder.transform(file_labels) for file_labels in y_train]
            else:
                y_train_enc = self.target_encoder.transform(y_train)

            # 创建模型实例（检查输出维度）
            num_classes = len(unique_labels)
            logger.info(f"===== 模型输出维度核心检查 =====")
            logger.info(f"根据标签计算的num_classes: {num_classes}")  # 关键：检查是否为5

            model_instance = self.model_factory(parameters_combination)
            
            # 检查模型实际输出维度
            if hasattr(model_instance, 'output_shape'):
                # 对于Keras模型，输出形状通常是 (None, ..., num_classes)
                model_output_dim = model_instance.output_shape[-1]
                logger.info(f"模型实际输出维度: {model_output_dim}")
            elif hasattr(model_instance, 'n_classes'):
                # 对于自定义模型，检查n_classes属性
                logger.info(f"模型定义的类别数: {model_instance.n_classes}")
            elif hasattr(model_instance, 'output_size'):
                # 适配DeepSoundBaseRNN的output_size属性
                logger.info(f"模型定义的输出尺寸: {model_instance.output_size}")
            else:
                logger.warning("无法检测模型输出维度，请检查模型定义")
            logger.info(f"=========================")

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

            # 传入回调函数进行训练
            funnel.fit(X_train, y_train_enc, callbacks=self.callbacks)

            # 量化处理
            if self.quantization:
                for ix_layer, layer in enumerate(funnel.model.model.layers):
                    w = layer.get_weights()
                    w = [i.astype(self.quantization) for i in w]
                    funnel.model.model.layers[ix_layer].set_weights(w)
                logger.info(f'量化处理已应用，类型：{str(self.quantization)}')

            # 模型预测与评估
            fold_window_acc = []  # 当前折的窗口级准确率
            for test_signal_key in test_fold_keys:
                # 所有片段都输出详细标签
                logger.info("="*50)
                logger.info(f"===== 开始处理片段 {test_signal_key} =====")
                logger.info("="*50)

                X_test = [self.X[test_signal_key]] if self.manage_sequences else self.X[test_signal_key]
                # 获取原始标签（填充前）
                y_test_raw = self.y[test_signal_key]
                y_test_flat = np.ravel(y_test_raw)
                original_length = len(y_test_flat)
                
                # 打印填充前的原始标签
                logger.info(f"测试片段 {test_signal_key} - 填充前原始标签（编码前）:")
                self._print_labels_in_chunks(y_test_raw, chunk_size=20)
                
                # 编码原始标签用于对比
                y_test_enc = self.target_encoder.transform(y_test_flat)
                logger.info(f"测试片段 {test_signal_key} - 填充前编码标签:")
                self._print_labels_in_chunks(y_test_enc, chunk_size=20)

                # 明确指定不聚合，获取窗口级预测（可能经过填充）
                y_signal_pred = funnel.predict(X_test)
                processed_length = len(y_signal_pred)
                
                # 打印填充后的预测标签
                logger.info(f"测试片段 {test_signal_key} - 填充后预测标签:")
                self._print_labels_in_chunks(y_signal_pred, chunk_size=20)

                # 截断到原始长度
                truncated_length = min(processed_length, original_length)
                y_signal_pred_truncated = y_signal_pred[:truncated_length]
                
                # 打印截断后的预测标签
                logger.info(f"测试片段 {test_signal_key} - 截断后预测标签:")
                self._print_labels_in_chunks(y_signal_pred_truncated, chunk_size=20)

                # 输出标签处理信息
                logger.info(f"测试片段 {test_signal_key} - 标签长度信息：")
                logger.info(f"  原始长度（填充前）: {original_length}")
                logger.info(f"  填充后长度: {processed_length}")
                logger.info(f"  截断后长度: {truncated_length}")

                # 打印标签数量与分布
                logger.info(f"测试片段 {test_signal_key} - 预测编码标签数量: {len(y_signal_pred)}")
                logger.info(f"测试片段 {test_signal_key} - 真实编码标签数量: {len(y_test_enc)}")
                logger.info(f"测试片段 {test_signal_key} - 预测编码标签中的唯一值: {np.unique(y_signal_pred)}")
                
                # 计算分布
                pred_flat = y_signal_pred.flatten()
                unique_pred, counts_pred = np.unique(pred_flat, return_counts=True)
                pred_dist = dict(zip(unique_pred, counts_pred))
                logger.info(f"测试片段 {test_signal_key} 预测类别分布：{pred_dist}")

                # 检查预测概率分布（若模型支持）
                try:
                    if hasattr(funnel.model, 'predict_proba'):
                        # 获取特征用于概率预测
                        X_features = []
                        for feature in funnel.features:
                            X_features.append(feature.transform(X_test))
                        if not self.use_raw_data:
                            X_features = np.concatenate(X_features, axis=1)
                        
                        y_probs = funnel.model.predict_proba(X_features)
                        logger.info(f"===== 预测概率分布检查 =====")
                        logger.info(f"概率输出形状（样本数, 类别数）: {y_probs.shape}")  # 关键：检查是否为5
                        logger.info(f"第一个样本的概率分布: {y_probs[0][:5]}")  # 查看前5类概率
                        logger.info(f"=========================")
                except Exception as e:
                    logger.warning(f"获取预测概率失败: {e}")

                # 逆编码预测结果（已为窗口级）
                y_signal_pred_labels = self.target_encoder.inverse_transform(y_signal_pred_truncated)

                # 计算窗口级准确率：确保长度一致
                if len(y_signal_pred_labels) != len(y_test_flat):
                    logger.warning(f"测试片段 {test_signal_key} 标签长度不匹配：预测{len(y_signal_pred_labels)} vs 真实{len(y_test_flat)}，进行长度对齐")
                    if len(y_signal_pred_labels) < len(y_test_flat):
                        pad_length = len(y_test_flat) - len(y_signal_pred_labels)
                        y_signal_pred_labels = np.pad(
                            y_signal_pred_labels,
                            (0, pad_length),
                            mode='edge'
                        )
                    else:
                        y_signal_pred_labels = y_signal_pred_labels[:len(y_test_flat)]
                
                # 过滤填充类别4（只保留有效标签0-3）
                # 将标签编码为数字以便过滤
                y_test_enc = self.target_encoder.transform(y_test_flat)
                y_pred_enc = self.target_encoder.transform(y_signal_pred_labels)
                
                valid_mask = (y_test_enc != 4)
                if np.sum(valid_mask) == 0:
                    logger.warning("当前片段无有效标签（全为填充值）")
                    window_acc = 0.0
                else:
                    valid_preds = y_pred_enc[valid_mask]
                    valid_trues = y_test_enc[valid_mask]
                    window_acc = np.mean(valid_preds == valid_trues)
                
                fold_window_acc.append(window_acc)
                logger.info(f"测试片段 {test_signal_key} 窗口级准确率：{window_acc:.4f}")
                logger.debug(f"窗口级分类报告：\n{classification_report(y_test_flat, y_signal_pred_labels)}")

                # 保存预测结果
                signal_predictions[test_signal_key] = [y_test_flat, y_signal_pred_labels]

                logger.info("="*50)
                logger.info(f"===== 片段 {test_signal_key} 处理完成 =====")
                logger.info("="*50)

            # 记录当前折的窗口级平均准确率
            if fold_window_acc:
                mean_window_acc = np.mean(fold_window_acc)
                window_metrics.append(mean_window_acc)
                logger.info(f"折 {ix_fold} 窗口级平均准确率：{mean_window_acc:.4f}")

            # 每折结束释放资源
            logger.info(f"折 {ix_fold} 训练/测试完成，开始释放资源...")
            if hasattr(self, 'model'):
                del self.model
            if 'funnel' in locals():
                del funnel
            if 'model_instance' in locals():
                del model_instance
            del X_train, y_train, y_train_enc
            if 'X_augmented' in locals():
                del X_augmented, y_augmented
            tf.keras.backend.clear_session()
            gc.collect()
            try:
                for gpu in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.reset_memory_stats(gpu.name)
                logger.info("GPU内存统计已重置")
            except Exception as e:
                logger.warning(f"重置GPU内存统计失败：{e}")
            logger.info(f"折 {ix_fold} 资源释放完成\n")

        # 评估与保存结果
        logger.info('-' * 25)
        logger.info('Fold iterations finished !. Starting evaluation phase.')
        self.save_predictions(signal_predictions)

        # 输出窗口级整体指标
        if window_metrics:
            logger.info('-' * 25)
            logger.info('### 窗口级整体指标 ###')
            logger.info(f"各折窗口准确率：{[f'{acc:.4f}' for acc in window_metrics]}")
            logger.info(f"平均窗口准确率：{np.mean(window_metrics):.4f} ± {np.std(window_metrics):.4f}")

        unique_labels = list(set(np.concatenate([self.y[k] for k in self.y.keys()])))
        if self.no_event_class in unique_labels:
            unique_labels.remove(self.no_event_class)

        if is_grid_search:
            fold_metrics = self.evaluate(unique_labels=unique_labels, folds=folds, verbose=False)
        else:
            logger.info('-' * 50)
            logger.info('***** Classification results *****')
            fold_metrics = self.evaluate(unique_labels=unique_labels, folds=folds, verbose=True)

        return fold_metrics

    def _print_labels_in_chunks(self, labels, chunk_size=20):
        """分块打印长标签列表"""
        if isinstance(labels, (list, np.ndarray)):
            for i in range(0, len(labels), chunk_size):
                chunk = labels[i:i+chunk_size]
                logger.info(f"  窗口 [{i}-{min(i+chunk_size-1, len(labels)-1)}]: {chunk}")
        else:
            logger.info(f"  标签数据格式异常: {type(labels)}")

    def save_predictions(self, fold_labels_predictions):
        '''保存预测结果，新增窗口级标签文件'''
        # 保存窗口级标签与预测
        df_window = pd.DataFrame(columns=['segment', 'window_index', 'y_true', 'y_pred'])
        for segment in fold_labels_predictions.keys():
            y_true = fold_labels_predictions[segment][0]
            y_pred = fold_labels_predictions[segment][1]
            
            # 窗口级详情
            _df = pd.DataFrame({
                'segment': segment,
                'window_index': range(len(y_true)),
                'y_true': y_true,
                'y_pred': y_pred
            })
            df_window = pd.concat([df_window, _df], ignore_index=True)

            # 事件级转换与保存
            df_labels = windows2events(
                y_true, 
                self.window_width, 
                self.window_overlap,
                no_event_class=self.no_event_class
            )
            df_labels = df_labels[df_labels.label != self.no_event_class]
            df_labels.to_csv(os.path.join(self.path, f'{segment}_true.txt'),
                             sep='\t', header=False, index=False)

            df_predictions = windows2events(
                y_pred, 
                self.window_width, 
                self.window_overlap,
                no_event_class=self.no_event_class
            )
            df_predictions = df_predictions[df_predictions.label != self.no_event_class]
            df_predictions.to_csv(os.path.join(self.path, f'{segment}_pred.txt'),
                                  sep='\t', header=False, index=False)

        # 保存窗口级详情CSV
        df_window.to_csv(os.path.join(self.path, 'window_level_predictions.csv'), index=False)
        logger.info(f"窗口级预测结果已保存至 {self.path}/window_level_predictions.csv")

    def evaluate(self, unique_labels, folds, verbose=True):
        '''评估模型性能，区分事件级与窗口级'''
        target_files = glob(os.path.join(self.path, 'recording_*_true.txt'))
        final_metric = 'f_measure'
        fold_metrics_detail = {}
        fold_metrics = []

        for ix_fold, fold in folds.items():
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

            # 计算事件级指标
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

            # 保存指标
            metrics = {
                'segment_based_metrics': segment_based_metrics,
                'event_based_metrics': event_based_metrics
            }
            with open(os.path.join(self.path, f'experiment_metrics_fold_{ix_fold}.pkl'), 'wb') as handle:
                pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # 提取指标结果
            segment_metrics = segment_based_metrics.results_overall_metrics()
            event_metrics = event_based_metrics.results_overall_metrics()

            if verbose:
                logger.info('### 片段级指标 (fold %s) ###', ix_fold)
                logger.info(segment_based_metrics)
                logger.info('')
                logger.info('### 事件级指标 (fold %s) ###', ix_fold)
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
            logger.info('### 事件级整体指标 ###')
            logger.info(f'事件F1分数均值: {folds_mean}')
            logger.info(f'事件F1分数标准差: {folds_std}')
            logger.info('-' * 20)

        return folds_mean

    def set_model_output_path(self, n_fold, is_grid_search=False):
        '''设置模型输出路径'''
        output_logs_path = os.path.join(self.path, f'logs_fold_{n_fold}')
        output_model_checkpoint_path = os.path.join(self.path, f'model_checkpoints_fold_{n_fold}')

        if not is_grid_search:
            if os.path.exists(output_logs_path):
                raise FileExistsError('Model output logs path already exists!')
            if os.path.exists(output_model_checkpoint_path):
                raise FileExistsError('Model output checkpoints path already exists!')

        os.makedirs(output_logs_path, exist_ok=True)
        self.model.output_logs_path = output_logs_path

        os.makedirs(output_model_checkpoint_path, exist_ok=True)
        self.model.output_path_model_checkpoints = output_model_checkpoint_path


class Funnel:
    '''类似sklearn Pipeline的接口，用于并行特征处理'''
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

    def fit(self, X, y, callbacks=None):
        '''拟合特征并训练模型（新增callbacks参数）'''
        X_features = []
        for feature in self.features:
            logger.info(f'Processing the feature {feature.feature.__class__.__name__}.')
            X_features.append(feature.fit_transform(X, y))

        if not self.use_raw_data:
            X_features = np.concatenate(X_features, axis=1)

        logger.info('Training model ...')
        # 传递回调函数给模型训练
        self.model.fit(X_features, y, callbacks=callbacks or [])

    def predict(self, X):
        '''特征转换并预测'''
        X_features = []
        for feature in self.features:
            X_features.append(feature.transform(X))

        if not self.use_raw_data:
            X_features = np.concatenate(X_features, axis=1)

        # 调用模型预测时不聚合，返回窗口级结果
        return self.model.predict(X_features, aggregate=False)