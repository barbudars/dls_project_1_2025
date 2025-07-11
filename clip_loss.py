#!/usr/bin/env python3
"""
CLIP Loss Module
Функции для работы с CLIP loss и кодированием
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from utils import prepare_image_for_clip

class CLIPLoss(nn.Module):
    def __init__(self, clip_model, device='cuda'):
        super().__init__()
        self.clip_model = clip_model
        self.device = device
        
        # Промпты для аниме стиля
        self.source_prompts = [
            "a realistic human face",
            "a photograph of a person",
            "realistic portrait"
        ]
        
        self.target_prompts = [
            "anime character face with large eyes",
            "japanese anime style portrait", 
            "manga character drawing",
            "anime girl with colorful hair"
        ]
        
        # Предварительное кодирование промптов
        self.source_embeddings = self._encode_prompts(self.source_prompts)
        self.target_embeddings = self._encode_prompts(self.target_prompts)
        
    def _encode_prompts(self, prompts):
        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)
            embeddings = self.clip_model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.mean(dim=0, keepdim=True)  # Усредняем все промпты
    
    def directional_loss(self, source_images, target_images):
        """Directional CLIP loss как в оригинальной статье"""
        # Подготовка изображений для CLIP
        source_clip = torch.stack([prepare_image_for_clip(img) for img in source_images])
        target_clip = torch.stack([prepare_image_for_clip(img) for img in target_images])
        
        # Получение эмбеддингов изображений
        source_img_embeddings = self.clip_model.encode_image(source_clip)
        target_img_embeddings = self.clip_model.encode_image(target_clip)
        
        # Нормализация
        source_img_embeddings = source_img_embeddings / source_img_embeddings.norm(dim=-1, keepdim=True)
        target_img_embeddings = target_img_embeddings / target_img_embeddings.norm(dim=-1, keepdim=True)
        
        # Направление в пространстве изображений
        img_direction = target_img_embeddings - source_img_embeddings
        img_direction = img_direction / img_direction.norm(dim=-1, keepdim=True)
        
        # Направление в пространстве текста
        text_direction = self.target_embeddings - self.source_embeddings
        text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)
        
        # Directional loss - максимизируем косинусное сходство направлений
        directional_similarity = torch.cosine_similarity(img_direction, text_direction, dim=-1)
        
        return 1 - directional_similarity.mean()
    
    def identity_loss(self, source_images, target_images):
        """Identity loss для сохранения семантики лиц"""
        source_clip = torch.stack([prepare_image_for_clip(img) for img in source_images])
        target_clip = torch.stack([prepare_image_for_clip(img) for img in target_images])
        
        source_embeddings = self.clip_model.encode_image(source_clip)
        target_embeddings = self.clip_model.encode_image(target_clip)
        
        source_embeddings = source_embeddings / source_embeddings.norm(dim=-1, keepdim=True)
        target_embeddings = target_embeddings / target_embeddings.norm(dim=-1, keepdim=True)
        
        similarity = torch.cosine_similarity(source_embeddings, target_embeddings, dim=-1)
        return 1 - similarity.mean()

def encode_text_prompts(clip_model, prompts, device='cuda'):
    """Кодирует текстовые промпты в CLIP embeddings"""
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        text_embeddings = clip_model.encode_text(tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings

def encode_images(clip_model, images, device='cuda'):
    """Кодирует изображения в CLIP embeddings"""
    # Преобразование StyleGAN изображений для CLIP
    # StyleGAN выдает [-1, 1], CLIP ожидает [0, 1] с нормализацией
    if images.min() < 0:
        images_norm = (images + 1) / 2  # [-1, 1] -> [0, 1]
    else:
        images_norm = images
    
    # Resize для CLIP (224x224)
    images_resized = F.interpolate(images_norm, size=224, mode='bilinear', align_corners=False)
    
    # CLIP нормализация
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(1, 3, 1, 1)
    images_normalized = (images_resized - mean) / std
    
    # Получение CLIP embeddings
    image_embeddings = clip_model.encode_image(images_normalized)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    
    return image_embeddings

def compute_clip_loss(clip_model, images_train, images_frozen, target_prompt, source_prompt="a face", device='cuda'):
    """Вычисляет CLIP loss между обученными и замороженными изображениями"""
    
    # Кодируем промпты
    source_embeddings = encode_text_prompts(clip_model, [source_prompt], device)
    target_embeddings = encode_text_prompts(clip_model, [target_prompt], device)
    
    # Кодируем изображения
    train_embeddings = encode_images(clip_model, images_train, device)
    frozen_embeddings = encode_images(clip_model, images_frozen, device)
    
    # Directional CLIP loss (как в StyleGAN-NADA)
    image_direction = train_embeddings - frozen_embeddings
    image_direction = image_direction / image_direction.norm(dim=-1, keepdim=True)
    
    text_direction = target_embeddings - source_embeddings
    text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)
    
    directional_similarity = torch.cosine_similarity(image_direction, text_direction, dim=-1)
    directional_loss = 1 - directional_similarity.mean()
    
    # Target loss (дополнительно для более сильной адаптации)
    target_similarity = torch.cosine_similarity(train_embeddings, target_embeddings, dim=-1)
    target_loss = -target_similarity.mean()
    
    # Комбинированная CLIP потеря
    total_clip_loss = 0.5 * directional_loss + 0.5 * target_loss
    
    return total_clip_loss, directional_similarity.mean().item(), target_similarity.mean().item()