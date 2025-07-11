#!/usr/bin/env python3
"""
StyleGAN-NADA Trainer Module
Основной тренер для адаптивного обучения
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
from clip_loss import compute_clip_loss, encode_text_prompts, encode_images
import clip

class AdvancedStyleGANNADATrainer:
    def __init__(self, generator_train, generator_frozen, clip_model, device, output_dir='results', log_dir='logs'):
        # Генераторы
        self.generator_train = generator_train.train()
        self.generator_frozen = generator_frozen.eval()
        
        # Замораживаем G_frozen
        for param in self.generator_frozen.parameters():
            param.requires_grad = False
            
        self.clip_model = clip_model
        self.device = device
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.history = None
        
        # Создаем директории
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.log = self.setup_logging(log_dir, verbose=True)
    
    def setup_logging(self, log_dir, verbose=False):
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
    
    def compute_identity_loss(self, images_train, images_frozen):
        """Потеря для сохранения структуры оригинальных изображений"""
        return F.mse_loss(images_train, images_frozen)
    
    def generate_images(self, generator, z):
        """Генерирует изображения с помощью StyleGAN2"""
        with torch.no_grad() if generator == self.generator_frozen else torch.enable_grad():
            if hasattr(generator, 'forward'):
                images = generator(z, None)
            else:
                images = generator(z)
        return images
    
    def adaptive_train(self, num_steps, target_prompt, initial_lr=5e-4, batch_size=8, source_prompt="a face"):
        """
        УЛУЧШЕННОЕ обучение с автоматическими шедулерами и улучшенными адаптивными весами
        ИСПРАВЛЕНО: batch_size=8 согласно StyleGAN-NADA статье
        УЛУЧШЕНО: initial_lr=5e-4 для более плавного обучения
        """
        
        self.log(f"Запуск УЛУЧШЕННОГО адаптивного обучения StyleGAN-NADA:")
        self.log(f"УЛУЧШЕННЫЙ Initial Learning Rate: {initial_lr} (более плавное обучение)")
        self.log(f"УЛУЧШЕННЫЕ автоматические шедулеры: ✓")
        self.log(f"УЛУЧШЕННЫЕ адаптивные веса потерь: ✓")
        self.log(f"Early stopping для предотвращения потери идентичности: ✓")
        self.log(f"Шагов обучения: {num_steps}")
        self.log(f"Batch size: {batch_size} (не трогаем - N_w = 8 согласно статье)")
        
        # Оптимизатор с правильным learning rate
        optimizer = optim.Adam(self.generator_train.parameters(), lr=initial_lr)
        
        # Шедулер для learning rate (автоматическая регулировка)
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.8,        # Более медленное уменьшение LR
                patience=200,      # Увеличенное терпение
                min_lr=1e-5,       # Повышенный минимальный LR
                verbose=True
            )
        except TypeError:
            # Для новых версий PyTorch без verbose
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.8,        # Более медленное уменьшение LR
                patience=200,      # Увеличенное терпение
                min_lr=1e-5,       # Повышенный минимальный LR
            )
        
        # История обучения
        history = {
            'total_loss': [],
            'clip_loss': [],
            'identity_loss': [],
            'directional_similarity': [],
            'target_similarity': [],
            'learning_rate': [],
            'lambda_identity': []
        }
        
        # УЛУЧШЕННЫЕ адаптивные веса для лучшего баланса стиля и идентичности
        class ImprovedAdaptiveWeights:
            def __init__(self):
                self.clip_loss_history = []
                self.identity_loss_history = []
                self.lambda_identity = 0.25  # УВЕЛИЧЕНО: начальное значение для лучшего сохранения идентичности
                self.clip_weight = 1.0
                self.early_stop_counter = 0
                self.best_loss = float('inf')
                
            def update_weights(self, step, clip_loss, identity_loss, total_steps):
                self.clip_loss_history.append(clip_loss)
                self.identity_loss_history.append(identity_loss)
                
                # Прогрессивное уменьшение identity loss (более медленное)
                progress = step / total_steps
                
                # УЛУЧШЕНО: Более медленное уменьшение с высоким минимумом
                # Квадратичное уменьшение вместо линейного для более плавного перехода
                decay_factor = (1 - progress) ** 1.5  # Более медленное уменьшение
                self.lambda_identity = max(0.08, 0.25 * decay_factor)  # Минимум 0.08 вместо 0.01
                
                # НОВОЕ: Early stopping на основе Identity Loss
                current_total_loss = clip_loss + self.lambda_identity * identity_loss
                
                # Если Identity Loss растёт слишком быстро - увеличиваем её вес
                if len(self.identity_loss_history) > 100:
                    recent_identity_trend = np.mean(self.identity_loss_history[-20:]) - np.mean(self.identity_loss_history[-100:-80])
                    if recent_identity_trend > 0.1:  # Identity Loss растёт быстро
                        self.lambda_identity = min(0.4, self.lambda_identity * 1.2)
                        print(f"  Identity Loss растёт! Увеличиваем lambda_identity до {self.lambda_identity:.3f}")
                
                # Адаптивное на основе соотношения потерь (более консервативное)
                if len(self.clip_loss_history) > 50:
                    recent_clip = np.mean(self.clip_loss_history[-50:])
                    recent_identity = np.mean(self.identity_loss_history[-50:])
                    
                    # Более консервативная адаптация
                    if recent_clip > recent_identity * 3:  # Увеличен порог
                        self.lambda_identity = min(0.5, self.lambda_identity * 1.05)  # Меньший множитель
                    elif recent_identity > recent_clip * 3:  # Увеличен порог
                        self.lambda_identity = max(0.08, self.lambda_identity * 0.95)  # Меньший множитель
                
                # НОВОЕ: Проверка на early stopping
                if current_total_loss < self.best_loss:
                    self.best_loss = current_total_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                return self.lambda_identity, self.clip_weight, self.early_stop_counter
        
        adaptive_weights = ImprovedAdaptiveWeights()
        
        # УЛУЧШЕННЫЙ основной цикл обучения с early stopping
        best_loss = float('inf')
        early_stop_patience = 300  # НОВОЕ: Терпение для early stopping
        
        for step in tqdm(range(num_steps), desc="StyleGAN-NADA (Improved)"):
            # Используем правильный batch_size=8 (не трогаем)
            z = torch.randn(batch_size, self.generator_train.z_dim, device=self.device)
            
            # Генерируем изображения
            self.generator_train.train()
            images_train = self.generate_images(self.generator_train, z)
            images_frozen = self.generate_images(self.generator_frozen, z)
            
            # Вычисляем потери
            clip_loss, directional_sim, target_sim = compute_clip_loss(
                self.clip_model, images_train, images_frozen, target_prompt, source_prompt, self.device
            )
            identity_loss = self.compute_identity_loss(images_train, images_frozen)
            
            # УЛУЧШЕННЫЕ адаптивные веса
            lambda_identity, clip_weight, early_stop_counter = adaptive_weights.update_weights(
                step, clip_loss.item(), identity_loss.item(), num_steps
            )
            
            # НОВОЕ: Early stopping если Identity Loss растёт слишком долго
            if early_stop_counter > early_stop_patience:
                self.log(f"\nEarly stopping на шаге {step}: Identity Loss не улучшается {early_stop_patience} шагов")
                self.log(f"Лучшая общая потеря: {adaptive_weights.best_loss:.4f}")
                break
            
            # Общая потеря с адаптивными весами
            total_loss = clip_weight * clip_loss + lambda_identity * identity_loss
            
            # Оптимизация
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.generator_train.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Обновляем шедулер
            scheduler.step(total_loss.item())
            
            # Сохраняем историю
            current_lr = optimizer.param_groups[0]['lr']
            history['total_loss'].append(total_loss.item())
            history['clip_loss'].append(clip_loss.item())
            history['identity_loss'].append(identity_loss.item())
            history['directional_similarity'].append(directional_sim)
            history['target_similarity'].append(target_sim)
            history['learning_rate'].append(current_lr)
            history['lambda_identity'].append(lambda_identity)
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
            
            # Логирование
            if step % 50 == 0:
                self.log(f"\nШаг {step}:")
                self.log(f"  Total Loss: {total_loss.item():.4f}")
                self.log(f"  CLIP Loss: {clip_loss.item():.4f}")
                self.log(f"  Identity Loss: {identity_loss.item():.4f}")
                self.log(f"  Directional Sim: {directional_sim:.4f}")
                self.log(f"  Target Sim: {target_sim:.4f}")
                self.log(f"  Lambda Identity: {lambda_identity:.3f} (adaptive)")
                self.log(f"  Learning Rate: {current_lr:.6f}")
                self.log(f"  Batch Size: {batch_size} (правильный)")
                
                # Визуализация промежуточных результатов
                if step % 200 == 0:
                    self.visualize_progress(z[0:1], target_prompt, step)
        
        self.log(f"\nАдаптивное обучение завершено!")
        self.log(f"Финальный Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        self.log(f"Финальный Lambda Identity: {lambda_identity:.3f}")
        
        self.history = history
        return history
    
    def visualize_progress(self, z, target_prompt, step):
        """Визуализирует прогресс обучения"""
        self.generator_train.eval()
        
        with torch.no_grad():
            img_train = self.generate_images(self.generator_train, z)
            img_frozen = self.generate_images(self.generator_frozen, z)
            
            def tensor_to_image(tensor):
                img = tensor[0].cpu()
                if img.min() < 0:
                    img = (img + 1) / 2
                img = img.permute(1, 2, 0).numpy()
                return np.clip(img, 0, 1)
            
            img_train_np = tensor_to_image(img_train)
            img_frozen_np = tensor_to_image(img_frozen)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(img_frozen_np)
            plt.title("G_frozen (Оригинал)")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img_train_np)
            plt.title(f"G_train (Шаг {step})")
            plt.axis('off')
            
            plt.suptitle(f"Адаптация к стилю: '{target_prompt}'", fontsize=14)
            plt.tight_layout()
            plt.show()
        
        self.generator_train.train()
    
    def plot_losses(self):
        """График потерь"""
        if not self.history:
            self.log("Нет данных о потерях. Запустите сначала adaptive_train()")
            return
            
        steps = range(len(self.history['total_loss']))
        
        plt.figure(figsize=(15, 10))
        
        # Общая потеря
        plt.subplot(2, 3, 1)
        plt.plot(steps, self.history['total_loss'])
        plt.title('Общая потеря')
        plt.xlabel('Шаг')
        plt.ylabel('Потеря')
        
        # CLIP потеря
        plt.subplot(2, 3, 2)
        plt.plot(steps, self.history['clip_loss'])
        plt.title('CLIP Loss')
        plt.xlabel('Шаг')
        plt.ylabel('Потеря')
        
        # Identity потеря
        plt.subplot(2, 3, 3)
        plt.plot(steps, self.history['identity_loss'])
        plt.title('Identity Loss')
        plt.xlabel('Шаг')
        plt.ylabel('Потеря')
        
        # Directional similarity
        plt.subplot(2, 3, 4)
        plt.plot(steps, self.history['directional_similarity'])
        plt.title('Directional Similarity')
        plt.xlabel('Шаг')
        plt.ylabel('Сходство')
        
        # Learning rate
        plt.subplot(2, 3, 5)
        plt.plot(steps, self.history['learning_rate'])
        plt.title('Learning Rate')
        plt.xlabel('Шаг')
        plt.ylabel('LR')
        plt.yscale('log')
        
        # Lambda identity
        plt.subplot(2, 3, 6)
        plt.plot(steps, self.history['lambda_identity'])
        plt.title('Lambda Identity (adaptive)')
        plt.xlabel('Шаг')
        plt.ylabel('Вес')
        
        plt.tight_layout()
        plt.show()