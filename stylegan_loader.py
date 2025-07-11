#!/usr/bin/env python3
"""
StyleGAN2 Loader Module
Функции для загрузки и работы с StyleGAN2 моделями
"""

import os
import pickle
import requests
import subprocess
import sys
from tqdm import tqdm
import torch
import torch.nn as nn

def install_stylegan2_dependencies():
    """Устанавливает зависимости для работы с официальным StyleGAN2"""
    try:
        # Установка ninja для компиляции CUDA extensions
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja"])
        
        # Клонируем официальный репозиторий StyleGAN2-ADA-PyTorch
        if not os.path.exists("stylegan2-ada-pytorch"):
            subprocess.run(["git", "clone", "https://github.com/NVlabs/stylegan2-ada-pytorch.git"], 
                          check=True)
            print("StyleGAN2-ADA-PyTorch репозиторий клонирован")
        
        # Добавляем в sys.path
        if "stylegan2-ada-pytorch" not in sys.path:
            sys.path.insert(0, "stylegan2-ada-pytorch")
        
        return True
        
    except Exception as e:
        print(f"Ошибка установки: {e}")
        print("Будет использована упрощенная реализация")
        return False

def download_stylegan2_ffhq():
    """Загружает официальную предобученную модель StyleGAN2-FFHQ"""
    
    model_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    model_path = "pretrained/stylegan2-ffhq-1024x1024.pkl"
    
    os.makedirs("pretrained", exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"Модель уже загружена: {model_path}")
        return model_path
    
    print(f"Загрузка StyleGAN2-FFHQ (размер: ~600MB)...")
    print(f"URL: {model_url}")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Загрузка StyleGAN2") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        print(f"StyleGAN2-FFHQ успешно загружен: {model_path}")
        print(f"Размер: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        return model_path
        
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return None

def load_stylegan2_generator(model_path, device='cuda'):
    """Загружает StyleGAN2 генератор из .pkl файла"""
    
    if not os.path.exists(model_path):
        print(f"Файл не найден: {model_path}")
        return None
    
    print(f"Загрузка StyleGAN2 генератора...")
    
    try:
        # Загружаем модель
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Извлекаем генератор
        if 'G_ema' in data:
            generator = data['G_ema'].to(device)
            print("Загружен G_ema (exponentially averaged generator)")
        elif 'G' in data:
            generator = data['G'].to(device)
            print("Загружен основной генератор G")
        else:
            print("Генератор не найден в файле")
            return None
        
        generator.eval()
        
        # Информация о модели
        print(f"Информация о StyleGAN2:")
        print(f"Разрешение: {generator.img_resolution}x{generator.img_resolution}")
        print(f"Каналы: {generator.img_channels}")
        print(f"z_dim: {generator.z_dim}")
        print(f"w_dim: {generator.w_dim}")
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in generator.parameters())
        print(f"Параметров: {total_params:,}")
        
        return generator
        
    except Exception as e:
        print(f"Ошибка загрузки генератора: {e}")
        return None

def load_generator(model_path, device=None):
    """Универсальная загрузка генератора: поддержка .pth (PyTorch checkpoint) и .pkl (NVlabs)"""
    import os
    import torch
    import pickle
    import sys
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(model_path):
        print(f'[load_generator] Файл не найден: {model_path}')
        return None
    ext = os.path.splitext(model_path)[-1].lower()
    ada_path = os.path.join(os.getcwd(), 'stylegan2-ada-pytorch')
    if os.path.isdir(ada_path) and ada_path not in sys.path:
        sys.path.insert(0, ada_path)
        print(f'[load_generator] stylegan2-ada-pytorch добавлен в sys.path')
    if ext == '.pth':
        print('[load_generator] Обнаружен .pth (PyTorch checkpoint)')
        checkpoint = torch.load(model_path, map_location=device)
        # Путь к официальной модели
        stylegan2_path = 'pretrained/stylegan2-ffhq-1024x1024.pkl'
        if not os.path.exists(stylegan2_path):
            print('[load_generator] Официальная модель StyleGAN2-FFHQ не найдена, скачиваю...')
            stylegan2_path = download_stylegan2_ffhq()
        generator = load_stylegan2_generator(stylegan2_path, device)
        if generator is None:
            print('[load_generator] Ошибка загрузки официального генератора')
            return None
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
            print('[load_generator] Веса из .pth успешно загружены!')
        else:
            print("[load_generator] В checkpoint отсутствует 'generator_state_dict'")
            return None
        generator = generator.to(device)
        generator.eval()
        return generator
    elif ext == '.pkl':
        print('[load_generator] Обнаружен .pkl (NVlabs pickle)')
        generator = load_stylegan2_generator(model_path, device)
        if generator is None:
            print('[load_generator] Ошибка загрузки генератора из .pkl')
            return None
        return generator
    else:
        print('[load_generator] Неизвестный формат файла весов:', ext)
        return None

def download_official_stylegan2_model(model_name='ffhq'):
    """Загружает различные официальные модели StyleGAN2"""
    
    # Официальные предобученные модели
    official_models = {
        'ffhq': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl',
        'ffhq_256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl',
        'afhqcat': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl',
        'afhqdog': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl',
        'afhqwild': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl',
        'metfaces': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl'
    }
    
    if model_name not in official_models:
        print(f"Модель '{model_name}' не найдена. Доступные: {list(official_models.keys())}")
        return None
    
    url = official_models[model_name]
    model_path = f"pretrained/stylegan2-{model_name}.pkl"
    
    os.makedirs("pretrained", exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"Модель {model_name} уже загружена: {model_path}")
        return model_path
    
    print(f"Загрузка StyleGAN2-{model_name.upper()}...")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Загрузка {model_name}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        print(f"StyleGAN2-{model_name} успешно загружен: {model_path}")
        print(f"Размер: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        return model_path
        
    except Exception as e:
        print(f"Ошибка загрузки {model_name}: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return None

def load_official_stylegan2_generator(model_path, device='cuda'):
    """Загружает официальный StyleGAN2 генератор с детальной информацией"""
    
    if not os.path.exists(model_path):
        print(f"Файл не найден: {model_path}")
        return None
    
    print(f"Загрузка StyleGAN2 генератора из {model_path}...")
    
    try:
        # Загружаем модель
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Извлекаем генератор
        if 'G_ema' in data:
            generator = data['G_ema'].to(device)
            print("Загружен G_ema (exponentially averaged generator)")
        elif 'G' in data:
            generator = data['G'].to(device)
            print("Загружен основной генератор G")
        else:
            print("Генератор не найден в файле")
            return None
        
        generator.eval()
        
        # Детальная информация о модели
        print(f"\nИнформация о StyleGAN2:")
        print(f"  Разрешение: {generator.img_resolution}x{generator.img_resolution}")
        print(f"  Каналы: {generator.img_channels}")
        print(f"  Z dim: {generator.z_dim}")
        print(f"  W dim: {generator.w_dim}")
        print(f"  Количество слоев: {generator.num_ws}")
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in generator.parameters())
        trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print(f"  Общие параметры: {total_params:,}")
        print(f"  Обучаемые параметры: {trainable_params:,}")
        
        return generator
        
    except Exception as e:
        print(f"Ошибка загрузки генератора: {e}")
        return None