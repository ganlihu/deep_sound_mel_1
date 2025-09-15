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
from sklearn.model_selection import ParameterGrid
import sed_eval
import dcase_util
import tensorflow as tf

from chewbite_fusion.data.utils import windows2events
from chewbite_fusion.experiments import settings
from chewbite_fusion.experiments.utils import set_random_init


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

        self.path = os.path.join(settings.experiments_path, name, self.timestamp)
        os.makedirs(self.path, exist_ok=True)

        # 配置日志
        logger.handlers = []
        fileHandler = logging.FileHandler(f"{self.path}/experiment.log")
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        set_random_init()

    def run(self):
        ''' 运行实验并保存相关信息 '''
        self.X = self.X['zavalla2022']
        self.y = self.y['zavalla2022']

        # 验证片段ID解析
        for k in self.X.keys():
            try:
                seg_id = int(k.split('_')[1])
                logger.debug(f'Segment {k} parsed to ID {seg_id}')
            except:
                logger.error(f'Failed to parse segment ID from {k}')

        # 折叠划分
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
        ''' 使用特定参数执行k折交叉验证 '''
        signal_predictions = {}

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

            # 标签编码 + 核心修改3：过滤标签4（若存在）
            # 1. 展平训练标签
            flat_y_train = np.hstack(y_train) if self.manage_sequences else y_train
            # 2. 过滤异常标签4
            valid_flat_y_train = flat_y_train[flat_y_train != 4]
            invalid_count = len(flat_y_train) - len(valid_flat_y_train)
            if invalid_count > 0:
                logger.warning(f"训练数据中过滤掉 {invalid_count} 个标签4")

            # 3. 编码有效标签
            self.target_encoder = LabelEncoder()
            unique_labels = np.unique(valid_flat_y_train)
            # 额外检查：确保无标签4
            if 4 in unique_labels:
                unique_labels = unique_labels[unique_labels != 4]
                logger.error("训练标签中仍存在4，已强制过滤")
            self.target_encoder.fit(unique_labels)

            # 记录标签映射关系
            logger.info("标签映射关系：")
            for idx, label in enumerate(self.target_encoder.classes_):
                logger.info(f"类别编号 {idx} → 原始标签 '{label}'")

            # 编码训练标签
            if self.manage_sequences:
                y_train_enc = []
                for file_labels in y_train:
                    # 过滤每个文件中的标签4
                    valid_labels = [l for l in file_labels if l != 4]
                    y_train_enc.append(self.target_encoder.transform(valid_labels))
            else:
                valid_y_train = [l for l in y_train if l != 4]
                y_train_enc = self.target_encoder.transform(valid_y_train)

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

            # 模型预测
            for test_signal_key in test_fold_keys:
                X_test = [self.X[test_signal_key]] if self.manage_sequences else self.X[test_signal_key]
                y_signal_pred = funnel.predict(X_test)

                # 处理预测结果
                if self.manage_sequences:
                    pred_flat = np.ravel(y_signal_pred[0])
                else:
                    pred_flat = y_signal_pred.flatten()
                
                # 过滤非数值预测
                if pred_flat.dtype == 'object':
                    logger.warning(f"测试片段 {test_signal_key} 发现非数值预测结果，正在清理...")
                    cleaned = []
                    for item in pred_flat:
                        if isinstance(item, dict):
                            if 'label' in item and isinstance(item['label'], (int, float)):
                                cleaned.append(item['label'])
                            else:
                                logger.warning(f"丢弃无效字典预测: {item}")
                        elif isinstance(item, (int, float, np.number)):
                            cleaned.append(item)
                        else:
                            logger.warning(f"丢弃非数值预测: {item} (类型: {type(item)})")
                    pred_flat = np.array(cleaned, dtype=np.float32)

                # 核心修改4：过滤预测结果中的标签4
                pred_flat = pred_flat[pred_flat != 4]
                if len(pred_flat) < len(y_signal_pred.flatten()):
                    logger.info(f"测试片段 {test_signal_key} 过滤掉预测结果中的标签4")

                # 计算预测分布
                unique_pred, counts_pred = np.unique(pred_flat, return_counts=True)
                pred_dist = dict(zip(unique_pred, counts_pred))
                logger.info(f"测试片段 {test_signal_key} 预测类别分布：{pred_dist}")

                # 检查异常类别
                max_expected = len(self.target_encoder.classes_) - 1 if len(self.target_encoder.classes_) > 0 else -1
                invalid_preds = [p for p in unique_pred if p > max_expected]
                if invalid_preds:
                    for p in invalid_preds:
                        count = counts_pred[unique_pred == p][0]
                        logger.warning(f"测试片段 {test_signal_key} 发现异常类别 {p}，共出现 {count} 次")

                # 逆编码预测结果
                y_signal_pred_enc = np.ravel(y_signal_pred[0]) if self.manage_sequences else y_signal_pred.flatten()
                if y_signal_pred_enc.dtype == 'object':
                    y_signal_pred_enc = np.array([
                        item['label'] if isinstance(item, dict) and 'label' in item else item
                        for item in y_signal_pred_enc
                        if isinstance(item, (int, float, np.number, dict))
                    ], dtype=np.int32)
                # 再次过滤标签4
                y_signal_pred_enc = y_signal_pred_enc[y_signal_pred_enc != 4]
                y_signal_pred_labels = self.target_encoder.inverse_transform(y_signal_pred_enc)

                # 合并相邻相同的预测结果
                y_signal_pred_labels = self.merge_adjacent_labels(y_signal_pred_labels)

                # 保存预测结果
                y_test = self.y[test_signal_key]
                signal_predictions[test_signal_key] = [y_test, y_signal_pred_labels]

            # 每折结束释放资源
            logger.info(f"折 {ix_fold} 训练/测试完成，开始释放资源...")
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

    def merge_adjacent_labels(self, labels):
        '''合并相邻且相同的标签'''
        if len(labels) == 0:
            return labels
        merged = [labels[0]]
        for label in labels[1:]:
            if label == merged[-1]:
                continue
            merged.append(label)
        return np.array(merged)

    def save_predictions(self, fold_labels_predictions):
        ''' 保存预测结果并转换为事件格式 '''
        df = pd.DataFrame(columns=['segment', 'y_true', 'y_pred'])

        for segment in fold_labels_predictions.keys():
            y_true = fold_labels_predictions[segment][0]
            y_pred = fold_labels_predictions[segment][1]

            min_length = min(len(y_true), len(y_pred))
            if len(y_true) != len(y_pred):
                logger.warning(f"片段 {segment} 的y_true与y_pred长度不匹配（{len(y_true)} vs {len(y_pred)}），已截断到最短长度")
                y_true = y_true[:min_length]
                y_pred = y_pred[:min_length]

            _df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
            _df['segment'] = segment
            df = pd.concat([df, _df], ignore_index=True)

            df_labels = windows2events(y_true, self.window_width, self.window_overlap)
            df_labels = df_labels[df_labels.label != self.no_event_class]
            df_labels.to_csv(os.path.join(self.path, f'{segment}_true.txt'),
                             sep='\t', header=False, index=False)

            df_predictions = windows2events(y_pred, self.window_width, self.window_overlap)
            df_predictions = df_predictions[df_predictions.label != self.no_event_class]
            df_predictions.to_csv(os.path.join(self.path, f'{segment}_pred.txt'),
                                  sep='\t', header=False, index=False)

        df.to_csv(os.path.join(self.path, 'fold_labels_and_predictions.csv'), index=False)

    def evaluate(self, unique_labels, folds, verbose=True):
        ''' 评估模型性能 '''
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

            event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
                event_label_list=unique_labels,
                t_collar=settings.collar_value
            )

            for file_pair in data:
                event_based_metrics.evaluate(
                    reference_event_list=file_pair['reference_event_list'],
                    estimated_event_list=file_pair['estimated_event_list']
                )

            with open(os.path.join(self.path, f'experiment_metrics_fold_{ix_fold}.pkl'), 'wb') as handle:
                pickle.dump({'event_based_metrics': event_based_metrics}, handle, protocol=pickle.HIGHEST_PROTOCOL)

            event_metrics = event_based_metrics.results_overall_metrics()

            if verbose:
                logger.info('### Event based metrics (fold %s) ###', ix_fold)
                logger.info(event_based_metrics)
                logger.info('-' * 20)

            fold_metrics_detail[ix_fold] = {'event_score': event_metrics[final_metric]}
            fold_metrics.append(event_metrics[final_metric][final_metric])

        with open(os.path.join(self.path, 'experiment_overall_metrics.pkl'), 'wb') as handle:
            pickle.dump(fold_metrics_detail, handle, protocol=pickle.HIGHEST_PROTOCOL)

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