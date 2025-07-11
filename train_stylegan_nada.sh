 #!/bin/bash

# StyleGAN-NADA Training Script for Linux
# Запуск: ./train_stylegan_nada.sh

set -e  # Остановка при ошибке

echo "========================================"
echo "StyleGAN-NADA Training Script"
echo "Адаптация StyleGAN2-FFHQ к аниме стилю"
echo "========================================"
echo

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для логирования
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка системы
log_info "Проверка системы..."

# Проверка Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 не найден! Установите: sudo apt install python3 python3-pip"
    exit 1
fi

log_success "Python3 найден: $(python3 --version)"

# Проверка CUDA
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU найден:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    log_warning "NVIDIA GPU не найден. Будет использован CPU (медленнее)"
fi

# Создание виртуального окружения
log_info "Создание виртуального окружения..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_success "Виртуальное окружение создано"
else
    log_info "Виртуальное окружение уже существует"
fi

# Активация виртуального окружения
log_info "Активация виртуального окружения..."
source venv/bin/activate

# Обновление pip
log_info "Обновление pip..."
pip install --upgrade pip

# Установка зависимостей
log_info "Установка зависимостей..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    log_error "Ошибка установки зависимостей!"
    exit 1
fi

log_success "Зависимости установлены"

# Создание директорий
log_info "Создание директорий..."
mkdir -p pretrained
mkdir -p datasets
mkdir -p results
mkdir -p logs

log_success "Директории созданы"

# Проверка существования модели
if [ -f "stylegan_nada_anime_model_corrected.pth" ]; then
    log_info "Найдена предобученная модель: stylegan_nada_anime_model_corrected.pth"
    log_info "Размер: $(du -h stylegan_nada_anime_model_corrected.pth | cut -f1)"
else
    log_warning "Предобученная модель не найдена"
fi

# Функция для запуска обучения
run_training() {
    local num_steps=${1:-1000}
    local target_prompt=${2:-"anime style face"}
    local batch_size=${3:-8}
    local learning_rate=${4:-5e-4}
    local pretrained_model=${5:-""}
    
    log_info "Запуск обучения с параметрами:"
    log_info "  Шагов: $num_steps"
    log_info "  Целевой стиль: $target_prompt"
    log_info "  Batch size: $batch_size"
    log_info "  Learning rate: $learning_rate"
    if [ -n "$pretrained_model" ]; then
        log_info "  Предобученная модель: $pretrained_model"
    fi
    
    # Запуск Python скрипта
    local cmd="python3 train_stylegan_nada.py \
        --steps $num_steps \
        --target_prompt \"$target_prompt\" \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --output_dir results \
        --log_dir logs \
        --verbose"
    
    # Добавляем предобученную модель если указана
    if [ -n "$pretrained_model" ]; then
        cmd="$cmd --pretrained_model $pretrained_model"
    fi
    
    log_info "Выполнение команды: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        log_success "Обучение завершено успешно!"
    else
        log_error "Ошибка во время обучения!"
        exit 1
    fi
}

# Функция для быстрого теста
run_quick_test() {
    log_info "Запуск быстрого теста..."
    python3 train_stylegan_nada.py --quick_test --verbose
    
    if [ $? -eq 0 ]; then
        log_success "Быстрый тест завершен!"
    else
        log_error "Ошибка в быстром тесте!"
        exit 1
    fi
}

# Функция для генерации образцов
generate_samples() {
    local num_samples=${1:-8}
    local seed=${2:-42}
    
    log_info "Генерация $num_samples образцов с seed=$seed..."
    
    python3 train_stylegan_nada.py --generate_samples $num_samples --seed $seed --verbose
    
    if [ $? -eq 0 ]; then
        log_success "Образцы сгенерированы!"
    else
        log_error "Ошибка генерации образцов!"
        exit 1
    fi
}

# Функция для создания отчета
create_report() {
    log_info "Создание отчета..."
    
    cat > results/report.txt << EOF
StyleGAN-NADA Training Report
=============================

Дата: $(date)
Система: $(uname -a)
Python: $(python3 --version)
CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Не доступен")

Параметры обучения:
- Шагов: ${STEPS:-1000}
- Целевой стиль: ${TARGET_PROMPT:-"anime style face"}
- Batch size: ${BATCH_SIZE:-8}
- Learning rate: ${LEARNING_RATE:-5e-4}

Созданные файлы:
$(find results -name "*.png" -o -name "*.pth" | sort)

Логи обучения:
$(find logs -name "*.log" 2>/dev/null | sort || echo "Логи не найдены")

EOF

    log_success "Отчет создан: results/report.txt"
}

# Главное меню
show_menu() {
    echo
    echo "Выберите действие:"
    echo "1) Быстрый тест (без обучения)"
    echo "2) Полное обучение (1000 шагов)"
    echo "3) Короткое обучение (500 шагов)"
    echo "4) Обучение с предобученной моделью"
    echo "5) Генерация образцов"
    echo "6) Создать отчет"
    echo "7) Выход"
    echo
    read -p "Введите номер (1-7): " choice
    
    case $choice in
        1)
            run_quick_test
            ;;
        2)
            run_training 1000 "anime style face" 8 5e-4 ""
            ;;
        3)
            run_training 500 "anime style face" 8 5e-4 ""
            ;;
        4)
            if [ -f "stylegan_nada_anime_model_corrected.pth" ]; then
                run_training 500 "oil painting style" 8 5e-4 "stylegan_nada_anime_model_corrected.pth"
            else
                log_warning "Предобученная модель не найдена!"
                log_info "Сначала запустите полное обучение (опция 2)"
            fi
            ;;
        5)
            generate_samples 8 42
            ;;
        6)
            create_report
            ;;
        7)
            log_info "Выход..."
            exit 0
            ;;
        *)
            log_error "Неверный выбор!"
            show_menu
            ;;
    esac
}

# Проверка аргументов командной строки
if [ $# -eq 0 ]; then
    # Интерактивный режим
    show_menu
else
    # Режим командной строки
    case $1 in
        "test")
            run_quick_test
            ;;
        "train")
            STEPS=${2:-1000}
            TARGET_PROMPT=${3:-"anime style face"}
            BATCH_SIZE=${4:-8}
            LEARNING_RATE=${5:-5e-4}
            PRETRAINED_MODEL=${6:-""}
            run_training $STEPS "$TARGET_PROMPT" $BATCH_SIZE $LEARNING_RATE "$PRETRAINED_MODEL"
            ;;
        "samples")
            NUM_SAMPLES=${2:-8}
            SEED=${3:-42}
            generate_samples $NUM_SAMPLES $SEED
            ;;
        "report")
            create_report
            ;;
        *)
            echo "Использование: $0 [test|train|samples|report]"
            echo "  test                                    - быстрый тест"
            echo "  train [steps] [prompt] [batch] [lr] [model] - обучение"
            echo "  samples [num] [seed]                   - генерация образцов"
            echo "  report                                  - создать отчет"
            echo ""
            echo "Примеры:"
            echo "  $0 train 1000 'anime style' 8 5e-4"
            echo "  $0 train 500 'oil painting' 8 5e-4 stylegan_nada_anime_model_corrected.pth"
            exit 1
            ;;
    esac
fi

log_success "Скрипт завершен!"