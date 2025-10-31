import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import random
import numpy as np
import cv2


class SimCLR_V2_Augmentation:
    def __init__(self):
        pass

    def __call__(self, image):
        return self.augment(image)

    def augment(self, image):
        width, height = image.size

        # 水平翻转
        if random.random() > 0.3:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 竖直翻转
        if random.random() > 0.6:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # 强旋转
        if random.random() > 0.3:
            angle = random.randint(-60, 60)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

        # ========== 2. 强透视变换 ==========
        if random.random() > 0.3:
            image = self._aggressive_perspective_transform(image, width, height)

        # ========== 3. 强缩放 ==========
        if random.random() > 0.3:
            scale = random.uniform(0.7, 1.3)
            image = self._scale_image(image, scale, width, height)

        # ========== 4. 强模糊 ==========
        if random.random() > 0.4:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2)))

        # ========== 5. 强噪声 ==========
        if random.random() > 0.4:
            image = self._add_strong_noise(image)

        # ========== 6. 多次随机擦除 ==========
        if random.random() > 0.4:
            image = self._aggressive_erasing(image)

        image = image.convert("L").convert("RGB")
        return image

    def _aggressive_perspective_transform(self, image, width, height):
        img_np = np.array(image)
        distortion = random.uniform(0.2, 0.4)

        half_w = width // 2
        half_h = height // 2

        from_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        to_points = np.float32([
            [random.randint(-int(distortion * half_w), int(distortion * half_w)),
             random.randint(-int(distortion * half_h), int(distortion * half_h))],
            [width + random.randint(-int(distortion * half_w), int(distortion * half_w)),
             random.randint(-int(distortion * half_h), int(distortion * half_h))],
            [width + random.randint(-int(distortion * half_w), int(distortion * half_w)),
             height + random.randint(-int(distortion * half_h), int(distortion * half_h))],
            [random.randint(-int(distortion * half_w), int(distortion * half_w)),
             height + random.randint(-int(distortion * half_h), int(distortion * half_h))]
        ])

        M = cv2.getPerspectiveTransform(from_points, to_points)
        img_np = cv2.warpPerspective(img_np, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return Image.fromarray(img_np)

    def _scale_image(self, image, scale, width, height):
        """缩放"""
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

        return image

    def _add_strong_noise(self, image):
        """添加强噪声"""
        img_np = np.array(image).astype(np.float32)
        noise_level = random.uniform(20, 50)  # 更强的噪声
        noise = np.random.normal(0, noise_level, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255)
        img_np = (img_np > 128).astype(np.uint8) * 255
        return Image.fromarray(img_np.astype(np.uint8))

    def _aggressive_erasing(self, image):
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        num_patches = random.randint(2, 5)
        for _ in range(num_patches):
            patch_h = random.randint(h // 15, h // 6)
            patch_w = random.randint(w // 15, w // 6)
            y = random.randint(0, max(1, h - patch_h))
            x = random.randint(0, max(1, w - patch_w))

            fill_value = random.choice([0, 255])
            img_np[y:y + patch_h, x:x + patch_w] = fill_value

        return Image.fromarray(img_np)

class SimCLR_V2_Dataset(Dataset):

    def __init__(self, csv_file, transform=None, use_augment=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_augment = use_augment
        self.augmenter = SimCLR_V2_Augmentation()

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
