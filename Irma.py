from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import clip
import torch

class Irma:
    """This is the IRMA data set.

    Paper
    -----
    Lehmann, T. M., Schubert, H., Keysers, D., Kohnen, M., & Wein, B. B.,
    The IRMA code for unique classification of medical images,
    In, Medical Imaging 2003: PACS and Integrated Medical Information Systems:
    Design and Evaluation (pp. 440–451) (2003). : International Society for Optics and Photonics.

    https://www.kaggle.com/raddar/irma-xray-dataset
    """

    def __init__(self, root, *args, **kwargs):
        self.data_dir = Path(root)

    def load(self):
        self.train_labels_path = self.data_dir / "ImageCLEFmed2009_train_codes.02.csv"
        self.train_images_path = self.data_dir / "ImageCLEFmed2009_train.02/ImageCLEFmed2009_train.02"
        # print(self.train_labels_path)
        df = pd.read_csv(self.train_labels_path, delimiter=";")
        df.loc[:, "Path"] = df["image_id"].apply(self._get_image_path)
        df.loc[:, "irma_code"] = df["irma_code"].apply(lambda x: x.replace("-", ""))
        df.loc[:, "Technical Code"] = df["irma_code"].apply(self._get_technical_code)
        df.loc[:, "Imaging Modality"] = df["Technical Code"].apply(self._get_imaging_modality)
        df.loc[:, "Directional Code"] = df["irma_code"].apply(self._get_directional_code)
        df.loc[:, "Imaging Orientation"] = df["Directional Code"].apply(self._get_imaging_orientation)
        df.loc[:, "Anatomical Code"] = df["irma_code"].apply(self._get_anatomical_code)
        df.loc[:, "Body Region"] = df["Anatomical Code"].apply(self._get_body_region)
        self.df = df

    def load_image(self, path: str) -> Image:
        """Cache and load an image."""
        return Image.open(path).convert("RGB")

    def _get_image_path(self, image_id: str) -> str:
        return self.train_images_path / f"{image_id}.png"

    def _get_technical_code(self, irma_code: str) -> str:
        return irma_code[:3]

    def _get_imaging_modality(self, technical_code: str):
        first, second, third = technical_code
        first_categories = {"0": "unspecified",
                            "1": "x-ray",
                            "2": "sonography",
                            "3": "magnetic resonance measurements",
                            "4": "nuclear medicine",
                            "5": "optical imaging",
                            "6": "biophysical procedure",
                            "7": "others",
                            "8": "secondary digitalization"}
        if first in first_categories:
            return first_categories[first]
        return technical_code

    def _get_directional_code(self, irma_code: str) -> str:
        return irma_code[3:6]

    def _get_imaging_orientation(self, directional_code: str) -> str:
        first, second, third = directional_code
        result = directional_code
        if first == 0:
            return "unspecified"
        elif first == 1:
            if second == 1:
                return "posteroanterior"
            elif second == 2:
                return "anteroposterior"
        elif first == 2:
            if second == 1:
                return "lateral, right-left"
            elif second == 2:
                return "lateral, left-right"
        return result

    def _get_anatomical_code(self, irma_code: str) -> str:
        return irma_code[6:9]

    def _get_body_region(self, anatomical_code: str) -> str:
        first, second, third = anatomical_code
        # print(first)
        first_categories = {
            "1": "whole body",
            "2": "cranium",
            "3": "spine",
            "4": "upper extremity/arm",
            "5": "chest",
            "6": "breast",
            "7": "abdomen",
            "8": "pelvis",
            "9": "lower extremity"
        }
        if first in first_categories:
            if second == "5":
                chest_categories = {
                    "0": "chest",
                    "1": "chest/bones",
                    "2": "chest/lung",
                    "3": "chest/hilum",
                    "4": "chest/mediastinum",
                    "5": "chest/heart",
                    "6": "chest/diaphragm"
                }
                return chest_categories[second]
            return first_categories[first]
        return 'Unknown'


class ClassificationDataset(Dataset):
    def __init__(self, df, transform=None, idx_range=None):
        self.transform = transform
        # self.labels = labels
        first_categories = {
            "1": "whole body",
            "2": "cranium",
            "3": "spine",
            "4": "upper extremity/arm",
            "5": "chest",
            "6": "breast",
            "7": "abdomen",
            "8": "pelvis",
            "9": "lower extremity"
        }
        chest_categories = {
            "0": "chest",
            "1": "chest/bones",
            "2": "chest/lung",
            "3": "chest/hilum",
            "4": "chest/mediastinum",
            "5": "chest/heart",
            "6": "chest/diaphragm"
        }
        # df = pd.read_csv(csv_path)

        # --- 🔍 自定义过滤规则 ---
        cleaned_rows = []
        for _, row in df.iterrows():
            target_field = str(row['Body Region'])  # 转成字符串


            # 保留仅有单个标签，并且是合法数字的
            if (target_field in chest_categories.values()) or (target_field in first_categories.values()):
                # row['Target'] = int(split_targets[0])
                cleaned_rows.append(row)

        self.data = pd.DataFrame(cleaned_rows)
        # csvdata = pd.DataFrame(cleaned_rows)
        # self.data = csvdata.iloc[range(idx_range[0],idx_range[1])]
        # if labels is not None:
        #     max_target = self.data['Target'].max()
        #     assert max_target < len(labels), f"标签索引 {max_target} 超出 Labels 长度 {len(labels)}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(row['Path'])
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = str(row['Body Region'])
        # label = self.labels[label_idx]
        # image = image.numpy()
        # image2 = np.array(image)
        return image, label


# if __name__ == '__main__':
#     irma = Irma('/home/sda2/dzb/Datasets/IRMA/')
#     irma.load()
#     labels = irma.df
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load('ViT-B/32', device)  # 0.62
#     dataset = ClassificationDataset(labels, transform=preprocess)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#     for images,label in dataloader:
#         print(images.size())
#         print(label)
#         break
    # print(labels['Body Region'])
