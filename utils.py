#!/usr/bin/env python3
"""
Utility Functions Module
Вспомогательные функции для работы с изображениями и данными
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os
import glob
import urllib.request
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def tensor_to_pil(tensor):
    """Преобразование тензора [-1, 1] в PIL изображение"""
    with torch.no_grad():
        tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Убираем batch dimension
        array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)

def tensor_to_pil_safe(tensor):
    """Безопасное преобразование тензора [-1, 1] в PIL изображение (с полной защитой от градиентов)"""
    with torch.no_grad():
        if tensor.requires_grad:
            tensor = tensor.detach()
        tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Убираем batch dimension
        array = tensor.cpu().numpy().transpose(1, 2, 0)
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)

def tensor_to_numpy_safe(tensor):
    """Безопасное преобразование тензора в numpy массив (для использования в тренере)"""
    with torch.no_grad():
        if tensor.requires_grad:
            tensor = tensor.detach()
        img = tensor[0].cpu()
        if img.min() < 0:
            img = (img + 1) / 2
        img = img.permute(1, 2, 0).numpy()
        return np.clip(img, 0, 1)

def prepare_image_for_clip(image_tensor):
    """Подготовка изображения для CLIP"""
    # Преобразование из [-1, 1] в [0, 1]
    image_tensor = (image_tensor + 1) / 2
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # Resize к 224x224 для CLIP
    transform = transforms.Resize((224, 224), antialias=True)
    image_tensor = transform(image_tensor)
    
    # Нормализация для CLIP
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    return normalize(image_tensor)

def generate_samples(generator, num_samples=8, seed=42, device='cuda'):
    """Генерация образцов из генератора"""
    torch.manual_seed(seed)
    z = torch.randn(num_samples, generator.z_dim, device=device)
    with torch.no_grad():
        images = generator(z, None)
    return images, z

def generate_images(generator, num_images=4, seed=42, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    z = torch.randn(num_images, generator.z_dim, device=device)
    with torch.no_grad():
        images = generator(z, None)
    return images

def create_trainable_copy_with_layer_freezing(generator, device='cuda', adaptation_type='texture', target_domain='anime'):
    """
    Создает обучаемую копию генератора с адаптивным замораживанием слоев
    согласно StyleGAN-NADA статье:
    
    - Для texture-based изменений (anime): все слои разморожены
    - Для shape изменений: k = 2/3 от общего числа слоев  
    - Для animal изменений: k = 3 слоя
    - batch_size N_w = 8
    """
    
    print(f"Создание тренируемой копии с адаптивным замораживанием слоев")
    print(f"Тип адаптации: {adaptation_type}")
    print(f"Целевой домен: {target_domain}")
    
    # Создаем глубокую копию генератора
    trainable_generator = copy.deepcopy(generator)
    trainable_generator = trainable_generator.to(device)
    trainable_generator.train()
    
    # Определяем количество тренируемых слоев согласно статье
    total_layers = 0
    layer_names = []
    
    # Подсчитываем слои в synthesis network (основная часть для адаптации)
    synthesis_layers = []
    for name, param in trainable_generator.named_parameters():
        if 'synthesis' in name and ('weight' in name or 'bias' in name):
            synthesis_layers.append((name, param))
            total_layers += 1
            layer_names.append(name)
    
    print(f"Synthesis layers для адаптации: {total_layers}")
    
    # Определяем количество тренируемых слоев по типу адаптации
    if adaptation_type == 'texture':
        # Для изменений на основе текстуры - разрешаем модификацию всех слоев
        trainable_layers = total_layers
        print("Тип: Текстурные изменения (anime) - обучаем ВСЕ слои")
    elif adaptation_type == 'shape':
        # Для небольших изменений формы - k = 2/3 от общего числа слоев
        trainable_layers = max(1, int(total_layers * 2 / 3))
        print(f"Тип: Изменения формы - обучаем {trainable_layers} из {total_layers} слоев")
    elif adaptation_type == 'animal':
        # Для модификации животных - k = 3
        trainable_layers = min(3, total_layers)
        print(f"Тип: Модификация животных - обучаем {trainable_layers} из {total_layers} слоев")
    else:
        # По умолчанию для anime - текстурные изменения
        trainable_layers = total_layers
        print("По умолчанию: обучаем ВСЕ слои")
    
    # Сначала замораживаем все параметры
    for param in trainable_generator.parameters():
        param.requires_grad = False
    
    # Затем размораживаем только нужное количество synthesis слоев (начиная с последних)
    # Последние слои отвечают за детали высокого разрешения
    if adaptation_type == 'texture':
        # Для текстурных изменений - размораживаем ВСЕ synthesis слои
        for name, param in synthesis_layers:
            param.requires_grad = True
            print(f"  Разморозили synthesis слой: {name}")
    else:
        # Для других типов - размораживаем только последние слои
        for i, (name, param) in enumerate(reversed(synthesis_layers)):
            if i < trainable_layers:
                param.requires_grad = True
                print(f"  Разморозили synthesis слой: {name}")
            else:
                print(f"  Заморозили synthesis слой: {name}")
    
    # Подсчитываем итоговое количество тренируемых параметров
    trainable_count = sum(p.numel() for p in trainable_generator.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in trainable_generator.parameters())
    
    print(f"\nИтоговая статистика:")
    print(f"Тренируемые параметры: {trainable_count:,}")
    print(f"Общее количество параметров: {total_count:,}")
    print(f"Процент тренируемых: {100 * trainable_count / total_count:.1f}%")
    
    return trainable_generator

def determine_adaptation_type(target_prompt):
    """
    Определяет тип адаптации на основе целевого промпта
    согласно StyleGAN-NADA статье
    """
    target_lower = target_prompt.lower()
    
    # Текстурные изменения (все слои)
    texture_keywords = ['anime', 'oil painting', 'sketch', 'drawing', 'painting', 'art style', 'pixar', 'cartoon']
    
    # Изменения формы (2/3 слоев)  
    shape_keywords = ['werewolf', 'elf', 'zombie', 'old', 'young', 'smile', 'expression']
    
    # Модификация животных (3 слоя)
    animal_keywords = ['dog', 'cat', 'lion', 'tiger', 'bear', 'animal']
    
    for keyword in texture_keywords:
        if keyword in target_lower:
            return 'texture'
    
    for keyword in shape_keywords:
        if keyword in target_lower:
            return 'shape'
            
    for keyword in animal_keywords:
        if keyword in target_lower:
            return 'animal'
    
    # По умолчанию - текстурные изменения
    return 'texture'

class CustomImageDataset(Dataset):
    """Кастомный датасет для загрузки изображений"""
    
    def __init__(self, root_dir, target_size=256, transform=None):
        self.root_dir = root_dir
        self.target_size = target_size
        
        # Поддерживаемые форматы
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Найти все изображения
        self.image_paths = []
        if os.path.exists(root_dir):
            for ext in self.valid_extensions:
                pattern = os.path.join(root_dir, f"**/*{ext}")
                self.image_paths.extend(glob.glob(pattern, recursive=True))
                pattern = os.path.join(root_dir, f"**/*{ext.upper()}")
                self.image_paths.extend(glob.glob(pattern, recursive=True))
        
        self.image_paths = list(set(self.image_paths))  # Убираем дубликаты
        
        # Трансформации
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
            # Возвращаем пустое изображение
            return torch.zeros(3, self.target_size, self.target_size)

def download_anime_dataset():
    """Инструкции по загрузке аниме датасета"""
    dataset_url = "https://www.kaggle.com/datasets/splcher/animefacedataset"
    
    print("Инструкции по загрузке датасета аниме лиц:")
    print("1. Перейдите по ссылке:", dataset_url)
    print("2. Скачайте архив и распакуйте в папку 'datasets/anime_faces'")
    print("3. Структура должна быть: datasets/anime_faces/*.jpg")

def download_sample_images():
    """Загружает образцы изображений для тестирования"""
    os.makedirs('datasets/sample_images', exist_ok=True)
    
    # URLs для примеров изображений (можно заменить на свои)
    sample_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/RGBCube_b.png/256px-RGBCube_b.png"
    ]
    
    for i, url in enumerate(sample_urls):
        try:
            filename = f'datasets/sample_images/sample_{i}.png'
            urllib.request.urlretrieve(url, filename)
            print(f"Загружен образец: {filename}")
        except Exception as e:
            print(f"Ошибка при скачивании {url}: {e}")

def create_dataloader(dataset_path, batch_size=4, target_size=256):
    """Создает DataLoader для обучения"""
    if not os.path.exists(dataset_path):
        print(f"Папка {dataset_path} не существует!")
        print("Создаю образцы изображений...")
        download_sample_images()
        dataset_path = 'datasets/sample_images'
    
    dataset = CustomImageDataset(dataset_path, target_size=target_size)
    
    if len(dataset) == 0:
        print(f"В папке {dataset_path} нет изображений!")
        return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader