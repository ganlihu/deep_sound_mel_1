import random
import numpy as np

# 步骤1：自动提取的52个编号（已确认完整）
existing_segments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
                     39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

# 步骤2：随机打乱编号（使用与项目一致的随机种子确保可复现）
random.seed(24)  # 与settings中的random_seed保持一致
random.shuffle(existing_segments)

# 步骤3：划分5折（52 = 10*4 + 12，最后一折多2个样本）
k = 5
fold_size = len(existing_segments) // k  # 10
folds = {}

for i in range(k):
    start = i * fold_size
    # 最后一折包含剩余所有样本（12个）
    end = start + fold_size if i < k-1 else len(existing_segments)
    folds[str(i+1)] = existing_segments[start:end]

# 步骤4：打印划分结果
print("5折交叉验证划分（52个编号）：")
for fold_num, segments in folds.items():
    print(f"折 {fold_num}：{segments}（共{len(segments)}个）")

# 步骤5：验证完整性和唯一性
all_segments = []
for seg in folds.values():
    all_segments.extend(seg)

assert set(all_segments) == set(existing_segments), "折划分覆盖不完整！"
assert len(all_segments) == len(set(all_segments)), "折划分存在重复编号！"
print("\n验证通过：所有52个编号均被覆盖且无重复")