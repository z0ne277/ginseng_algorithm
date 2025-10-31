import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import random
import numpy as np
import cv2


class MoCo_V3_Augmentation:

    def __init__(self):
        pass

    def __call__(self, image):
        return self.augment(image)

    def augment(self, image):
        width, height = image.size

        # ========== 几何变换 ==========
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 旋转
        if random.random() > 0.4:
            angle = random.randint(-30, 30)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

        # ========== 透视变换 ==========
        if random.random() > 0.6:
            image = self._perspective_transform(image, width, height)

        # ========== 缩放 ==========
        if random.random() > 0.5:
            scale = random.uniform(0.85, 1.15)
            new_w = int(width * scale)
            new_h = int(height * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)

            if scale > 1:
                left = (new_w - width) // 2
                top = (new_h - height) // 2
                image = image.crop((left, top, left + width, top + height))
            else:
                new_image = Image.new('L', (width, height), 0)
                paste_x = (width - new_w) // 2
                paste_y = (height - new_h) // 2
                new_image.paste(image, (paste_x, paste_y))
                image = new_image

        # ========== 模糊 ==========
        if random.random() > 0.7:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

        # ========== 噪声 ==========
        if random.random() > 0.7:
            image = self._add_noise(image)

        image = image.convert("L").convert("RGB")
        return image

    def _perspective_transform(self, image, width, height):
        img_np = np.array(image)
        distortion = random.uniform(0.05, 0.15)
        half_w = width // 2
        half_h = height // 2

        from_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        to_points = np.float32([
            [random.randint(0, int(distortion * half_w)), random.randint(0, int(distortion * half_h))],
            [width - random.randint(0, int(distortion * half_w)), random.randint(0, int(distortion * half_h))],
            [width - random.randint(0, int(distortion * half_w)), height - random.randint(0, int(distortion * half_h))],
            [random.randint(0, int(distortion * half_w)), height - random.randint(0, int(distortion * half_h))]
        ])

        M = cv2.getPerspectiveTransform(from_points, to_points)
        img_np = cv2.warpPerspective(img_np, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return Image.fromarray(img_np)

    def _add_noise(self, image):
        img_np = np.array(image).astype(np.float32)
        noise = np.random.normal(0, random.uniform(8, 15), img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255)
        img_np = (img_np > 128).astype(np.uint8) * 255
        return Image.fromarray(img_np.astype(np.uint8))

class MoCo_V3_Dataset(Dataset):
    def __init__(self, csv_file, transform=None, use_augment=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_augment = use_augment
        self.augmenter = MoCo_V3_Augmentation()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        img = Image.open(img_path).convert("L").convert("RGB")

        if self.use_augment:
            img1 = self.augmenter(img)
            img2 = self.augmenter(img)
        else:
            img1 = img
            img2 = img

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2
