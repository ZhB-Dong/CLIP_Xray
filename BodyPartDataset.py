import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ClassificationDataset(Dataset):
    def __init__(self, csv_path, root_dir='', transform=None, labels=None, idx_range=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = labels

        df = pd.read_csv(csv_path)

        # --- 🔍 自定义过滤规则 ---
        cleaned_rows = []
        for _, row in df.iterrows():
            target_field = str(row['Target'])  # 转成字符串
            split_targets = target_field.strip().split()

            # 保留仅有单个标签，并且是合法数字的
            if len(split_targets) == 1 and split_targets[0].isdigit():
                row['Target'] = int(split_targets[0])
                cleaned_rows.append(row)

        # self.data = pd.DataFrame(cleaned_rows)
        csvdata = pd.DataFrame(cleaned_rows)
        self.data = csvdata.iloc[range(idx_range[0],idx_range[1])]
        if labels is not None:
            max_target = self.data['Target'].max()
            assert max_target < len(labels), f"标签索引 {max_target} 超出 Labels 长度 {len(labels)}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row['image_path'])
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label_idx = row['Target']
        label = self.labels[label_idx]
        # image = image.numpy()
        # image2 = np.array(image)
        return image, label
