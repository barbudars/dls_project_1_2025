#!/usr/bin/env python3
"""
StyleGAN-NADA Training Script
Запуск: python train_stylegan_nada.py [опции]
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from einops import rearrange
import clip
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import requests
from io import BytesIO
import zipfile
import copy
import pickle
import subprocess
import sys
import glob
import tarfile
import shutil
import urllib.request
import warnings
import json
from datetime import datetime

# Импорты для работы с данными
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

# Импорты из новых модулей
from stylegan_loader import (
    install_stylegan2_dependencies,
    download_stylegan2_ffhq,
    load_stylegan2_generator,
    download_official_stylegan2_model,
    load_official_stylegan2_generator
)
from utils import (
    tensor_to_pil_safe,
    generate_samples,
    create_trainable_copy_with_layer_freezing,
    determine_adaptation_type
)
from trainer import AdvancedStyleGANNADATrainer

warnings.filterwarnings('ignore')

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='StyleGAN-NADA Training')
    
    parser.add_argument('--steps', type=int, default=1000,
                       help='Количество шагов обучения (по умолчанию: 1000)')
    parser.add_argument('--target_prompt', type=str, default='anime style face',
                       help='Целевой стиль (по умолчанию: anime style face)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Размер батча (по умолчанию: 8)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate (по умолчанию: 5e-4)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Директория для результатов (по умолчанию: results)')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Директория для логов (по умолчанию: logs)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Путь к официальной модели StyleGAN2-FFHQ')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Путь к предобученной модели для продолжения обучения')
    parser.add_argument('--quick_test', action='store_true',
                       help='Запустить только быстрый тест')
    parser.add_argument('--generate_samples', type=int, default=0,
                       help='Генерировать образцы (количество)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed для воспроизводимости (по умолчанию: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='Подробный вывод')
    
    return parser.parse_args()

def setup_logging(log_dir, verbose=False):
    """Настройка логирования"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    def log(message, level='INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f'[{timestamp}] [{level}] {message}'
        
        # Вывод в консоль
        if verbose or level in ['ERROR', 'WARNING']:
            print(log_message)
        
        # Запись в файл
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    return log

def check_system():
    """Проверка системы"""
    log = setup_logging('logs', verbose=True)
    
    log("Проверка системы...")
    
    # Проверка PyTorch версии
    torch_version = torch.__version__
    log(f"PyTorch версия: {torch_version}")
    
    # Проверка CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Устройство: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"GPU: {gpu_name}")
        log(f"Память GPU: {gpu_memory:.1f} GB")
    else:
        log("CUDA недоступен, будет использован CPU")
    
    # Проверка Python версии
    python_version = sys.version_info
    log(f"Python версия: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    return device

# Функции загрузки StyleGAN2 теперь импортируются из stylegan_loader.py

def load_pretrained_nada_model(model_path, device='cuda'):
    """Загружает предобученную StyleGAN-NADA модель на базе официального StyleGAN2-ADA"""
    log = setup_logging('logs', verbose=True)
    
    if not os.path.exists(model_path):
        log(f"Файл не найден: {model_path}", 'ERROR')
        return None
    
    log(f"Загрузка предобученной StyleGAN-NADA модели...")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Для совместимости: ищем путь к официальной модели
        stylegan2_path = 'pretrained/stylegan2-ffhq-1024x1024.pkl'
        if not os.path.exists(stylegan2_path):
            log(f"Официальная модель StyleGAN2-FFHQ не найдена, скачиваю...")
            from stylegan_loader import download_stylegan2_ffhq
            stylegan2_path = download_stylegan2_ffhq()
        generator = load_stylegan2_generator(stylegan2_path, device)
        if generator is None:
            log("Ошибка загрузки официального генератора", 'ERROR')
            return None
        # Загружаем веса
        if 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
            log("Веса предобученной модели успешно загружены!")
        else:
            log("В checkpoint отсутствует 'generator_state_dict'", 'ERROR')
            return None
        generator.eval()
        return generator
    except Exception as e:
        log(f"Ошибка загрузки предобученной модели: {e}", 'ERROR')
        return None

# Utility функции теперь импортируются из utils.py

# Класс тренера теперь импортируется из trainer.py

def main():
    """Основная функция с новой модульной архитектурой"""
    args = parse_args()
    
    # Настройка логирования
    log = setup_logging(args.log_dir, args.verbose)
    
    log("Запуск StyleGAN-NADA обучения с новой модульной архитектурой...")
    log(f"Параметры: steps={args.steps}, target_prompt='{args.target_prompt}', batch_size={args.batch_size}, lr={args.learning_rate}")
    
    # Проверка системы
    device = check_system()
    
    # Быстрый тест
    if args.quick_test:
        log("Запуск быстрого теста...")
        
        # Проверка CUDA
        log(f"CUDA доступен: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"GPU: {torch.cuda.get_device_name()}")
            log(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Проверка CLIP
        try:
            log("Тестирование загрузки CLIP...")
            clip_model, _ = clip.load("ViT-B/32", device=device)
            log("✅ CLIP загружен успешно")
        except Exception as e:
            log(f"❌ Ошибка загрузки CLIP: {e}", 'ERROR')
        
        # Тестирование создания модели
        try:
            log("Тестирование создания модели...")
            test_generator = load_stylegan2_generator("pretrained/stylegan2-ffhq-1024x1024.pkl", device)
            if test_generator:
                log("✅ Модель создана успешно")
                
                # Тестирование генерации
                with torch.no_grad():
                    z = torch.randn(1, test_generator.z_dim, device=device)
                    img = test_generator(z)
                    log(f"✅ Генерация успешна! Размер: {img.shape}")
            else:
                log("❌ Ошибка создания модели", 'ERROR')
        except Exception as e:
            log(f"❌ Ошибка тестирования модели: {e}", 'ERROR')
        
        # Проверка предобученной модели
        if os.path.exists("stylegan_nada_anime_model_corrected.pth"):
            try:
                log("Тестирование предобученной модели...")
                pretrained_generator = load_pretrained_nada_model("stylegan_nada_anime_model_corrected.pth", device)
                if pretrained_generator:
                    log("✅ Предобученная модель загружена успешно")
                else:
                    log("❌ Ошибка загрузки предобученной модели", 'WARNING')
            except Exception as e:
                log(f"❌ Ошибка тестирования предобученной модели: {e}", 'WARNING')
        
        log("✅ Быстрый тест завершен!")
        return
    
    # Генерация образцов
    if args.generate_samples > 0:
        log(f"Генерация {args.generate_samples} образцов...")
        torch.manual_seed(args.seed)
        
        # Создаем фиктивные образцы для демонстрации
        os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
        
        for i in range(args.generate_samples):
            # Генерируем случайное изображение
            img = torch.randn(3, 256, 256)
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            
            # Сохраняем
            filename = os.path.join(args.output_dir, 'samples', f'sample_{i:02d}.png')
            Image.fromarray(img).save(filename)
        
        log(f"Сохранено {args.generate_samples} образцов в {args.output_dir}/samples/")
        return
    
    # Загрузка модели
    G = None
    
    # Сначала пробуем загрузить предобученную модель
    pretrained_model_path = args.pretrained_model
    
    # Если путь не указан, ищем стандартные файлы
    if not pretrained_model_path:
        possible_paths = [
            'stylegan_nada_anime_model_corrected.pth',
            'stylegan_nada_anime_model.pth',
            'results/stylegan_nada_anime_model.pth'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pretrained_model_path = path
                log(f"Найдена предобученная модель: {path}")
                break
    
    if pretrained_model_path:
        log(f"Попытка загрузки предобученной модели: {pretrained_model_path}")
        G = load_pretrained_nada_model(pretrained_model_path, device)
        if G is not None:
            log("Предобученная модель успешно загружена!")
        else:
            log("Ошибка загрузки предобученной модели", 'WARNING')
    
    # Если предобученная модель не загружена, загружаем официальную StyleGAN2
    if G is None:
        log("Загрузка официальной модели StyleGAN2-FFHQ...")
        model_path = args.model_path or download_stylegan2_ffhq()
        
        if model_path:
            G = load_stylegan2_generator(model_path, device)
            if G is None:
                log("Ошибка загрузки StyleGAN2", 'ERROR')
                return
        else:
            log("Ошибка загрузки StyleGAN2", 'ERROR')
            return
    
    # Загрузка CLIP
    log("Загрузка CLIP модели...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    log("CLIP модель загружена")
    
    # Инициализация тренера с новой архитектурой
    log("Инициализация StyleGAN-NADA тренера с новой модульной архитектурой...")
    adaptation_type = determine_adaptation_type(args.target_prompt)
    log(f"Определен тип адаптации: {adaptation_type}")
    
    generator_train_with_freezing = create_trainable_copy_with_layer_freezing(
        generator=G,
        device=device,
        adaptation_type=adaptation_type,
        target_domain='anime'
    )
    
    trainer = AdvancedStyleGANNADATrainer(
        generator_train=generator_train_with_freezing,
        generator_frozen=G,
        clip_model=clip_model,
        device=device,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )
    
    log("AdvancedStyleGANNADATrainer инициализирован с новой архитектурой!")
    
    # Запуск обучения
    log("Запуск обучения...")
    history = trainer.adaptive_train(
        num_steps=args.steps,
        target_prompt=args.target_prompt,
        initial_lr=args.learning_rate,
        batch_size=args.batch_size,
        source_prompt="a face"
    )
    
    # Анализ результатов
    log("Анализ результатов обучения...")
    trainer.plot_losses()
    
    # Сохранение модели
    log("Сохранение обученной модели...")
    model_filename = os.path.join(args.output_dir, 'stylegan_nada_anime_model.pth')
    torch.save({
        'generator_state_dict': trainer.generator_train.state_dict(),
        'history': trainer.history,
        'target_prompt': args.target_prompt,
        'source_prompt': "a face",
        'final_metrics': {
            'total_loss': trainer.history['total_loss'][-1] if trainer.history else None,
            'clip_loss': trainer.history['clip_loss'][-1] if trainer.history else None,
            'identity_loss': trainer.history['identity_loss'][-1] if trainer.history else None,
            'lambda_identity': trainer.history['lambda_identity'][-1] if trainer.history else None,
        }
    }, model_filename)
    
    log(f"Модель сохранена в '{model_filename}'")
    if trainer.history:
        log(f"Общее количество шагов обучения: {len(trainer.history['total_loss'])}")
        log(f"Финальная общая потеря: {trainer.history['total_loss'][-1]:.4f}")
        log(f"Финальная CLIP потеря: {trainer.history['clip_loss'][-1]:.4f}")
        log(f"Финальная Identity потеря: {trainer.history['identity_loss'][-1]:.4f}")
    
    log("✅ Обучение завершено с новой модульной архитектурой!")

if __name__ == "__main__":
    main()