# src/datasets/fedif.py
import json
import os
from PIL import Image
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class FedIFGoldenDataset(Dataset):
    def __init__(self, json_file, transform=None, tokenizer=None, max_length=40):
        """
        读取 FedIF 黄金验证集 JSON
        """
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        if not os.path.exists(json_file):
            logger.error(f"[FedIF] Golden set file not found: {json_file}")
        else:
            logger.info(f"[FedIF] Loading golden set from: {json_file}")
            with open(json_file, 'r') as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        caption = item['caption']

        # 1. 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image: {image_path}, error: {e}")
            # 返回一个黑图防止崩溃（极端情况）
            image = Image.new('RGB', (224, 224))

        # 2. 图像预处理 (Transform)
        if self.transform:
            image = self.transform(image)

        # 3. 文本分词 (Tokenizer) -> 转为 BERT Token IDs
        # 注意：这里必须与 MoME 模型训练时的 Tokenizer 保持一致 (通常是 BERT)
        if self.tokenizer:
            # 如果是 BERT Tokenizer
            if hasattr(self.tokenizer, 'encode_plus') or hasattr(self.tokenizer, '__call__'):
                caption = self.tokenizer(
                    caption,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )['input_ids'][0]
            else:
                # 备用逻辑
                pass

        return image, caption, idx # 返回 idx 方便调试