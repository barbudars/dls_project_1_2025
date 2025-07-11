"""
StyleGAN-NADA Package

Модульная реализация StyleGAN-NADA для адаптации генераторов изображений
к различным стилям с использованием CLIP.

Основные модули:
- stylegan_loader: Загрузка и инициализация StyleGAN2 моделей
- utils: Вспомогательные функции и утилиты
- clip_loss: CLIP loss функции и кодирование
- trainer: Основной тренер для адаптивного обучения

Пример использования:
    python train_stylegan_nada.py --steps 1000 --target_prompt "anime style face" --verbose
"""

__version__ = "1.0.0"
__author__ = "StyleGAN-NADA Team"
__description__ = "Модульная реализация StyleGAN-NADA для адаптации генераторов изображений"

# Основные экспорты
try:
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
        determine_adaptation_type,
        CustomImageDataset
    )

    from trainer import AdvancedStyleGANNADATrainer

    from clip_loss import (
        CLIPLoss,
        encode_text_prompts,
        encode_images,
        compute_clip_loss
    )
except ImportError:
    # Если модули не найдены, создаем пустые заглушки
    pass

__all__ = [
    # StyleGAN loader
    'install_stylegan2_dependencies',
    'download_stylegan2_ffhq',
    'load_stylegan2_generator',
    'download_official_stylegan2_model',
    'load_official_stylegan2_generator',
    
    # Utils
    'tensor_to_pil_safe',
    'generate_samples',
    'create_trainable_copy_with_layer_freezing',
    'determine_adaptation_type',
    'CustomImageDataset',
    
    # Trainer
    'AdvancedStyleGANNADATrainer',
    
    # CLIP Loss
    'CLIPLoss',
    'encode_text_prompts',
    'encode_images',
    'compute_clip_loss',
]