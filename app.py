import streamlit as st
import torch
import os
import subprocess
from stylegan_loader import load_generator
from utils import generate_images, tensor_to_pil_safe

st.set_page_config(page_title="StyleGAN-NADA Demo", layout="centered")
st.title("StyleGAN-NADA: Было / Стало (Anime StyleGAN-NADA)")

# --- Автоматическое скачивание stylegan2-ada-pytorch ---
if not os.path.isdir('stylegan2-ada-pytorch'):
    st.warning('Скачиваю stylegan2-ada-pytorch с GitHub...')
    subprocess.run(['git', 'clone', 'https://github.com/NVlabs/stylegan2-ada-pytorch.git'])
    st.success('stylegan2-ada-pytorch успешно скачан!')

# --- Параметры ---
weights_path = 'stylegan_nada_anime_model_corrected.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def get_generator():
    return load_generator(weights_path, device=device)

generator = get_generator()

if generator is None:
    st.error("Не удалось загрузить генератор. Проверьте веса и зависимости.")
    st.stop()

st.markdown("""
**Инструкция:**
- Выберите количество изображений и seed.
- Нажмите 'Сгенерировать'.
- Сначала будут показаны оригинальные лица (до адаптации), затем стилизованные (аниме).
""")

num_images = st.slider("Количество изображений", 1, 8, 4)
seed = st.number_input("Seed (для повторяемости)", min_value=0, max_value=999999, value=42, step=1)

if st.button("Сгенерировать"):
    with st.spinner("Генерация изображений..."):
        # 1. Генерируем латентные вектора
        torch.manual_seed(int(seed))
        z = torch.randn(num_images, generator.z_dim, device=device)
        # 2. Генерируем оригинальные лица (до адаптации)
        #    Для этого нужен генератор с официальными весами (FFHQ)
        #    Загружаем его через load_generator с официальным .pkl
        ffhq_pkl = 'pretrained/stylegan2-ffhq-1024x1024.pkl'
        if not os.path.exists(ffhq_pkl):
            st.info('Скачиваю официальные веса StyleGAN2-FFHQ...')
            from stylegan_loader import download_stylegan2_ffhq
            ffhq_pkl = download_stylegan2_ffhq()
        from stylegan_loader import load_stylegan2_generator
        generator_ffhq = load_stylegan2_generator(ffhq_pkl, device)
        originals = generator_ffhq(z, None).detach().cpu()
        # 3. Генерируем стилизованные (аниме) лица
        stylized = generator(z, None).detach().cpu()
        st.subheader("Было (оригинал FFHQ):")
        cols1 = st.columns(num_images)
        for i in range(num_images):
            img = tensor_to_pil_safe(originals[i])
            cols1[i].image(img, caption=f"Оригинал {i+1}", use_container_width=True)
        st.subheader("Стало (аниме-стиль):")
        cols2 = st.columns(num_images)
        for i in range(num_images):
            img = tensor_to_pil_safe(stylized[i])
            cols2[i].image(img, caption=f"Anime {i+1}", use_container_width=True) 