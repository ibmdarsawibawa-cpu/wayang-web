import io
from collections import Counter

import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

MAKS_FOTO = 5
SISI_MAKS = 1024   # foto diperkecil dulu agar cepat & hemat RAM

st.set_page_config(page_title='Deteksi Motif Ukiran Wayang Kulit Bali', layout='wide')
st.title('Deteksi Motif Ukiran Wayang Kulit Bali')
st.caption(f'Unggah hingga {MAKS_FOTO} foto wayang kulit untuk mendeteksi motif ukirannya.')


@st.cache_resource
def muat_model():
    return YOLO('best.pt')


model = muat_model()


@st.cache_data(show_spinner=False, max_entries=20)
def deteksi(data: bytes, conf: float):
    # perbaiki rotasi EXIF, ubah ke RGB, perkecil
    gambar = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
    gambar.thumbnail((SISI_MAKS, SISI_MAKS))

    # kirim objek PIL langsung agar warna tidak terbalik
    hasil = model.predict(gambar, conf=conf, imgsz=640, verbose=False)[0]
    gambar_hasil = hasil.plot()[:, :, ::-1]   # BGR ke RGB
    jumlah = Counter(model.names[int(c)] for c in hasil.boxes.cls)
    return gambar, gambar_hasil, dict(jumlah)


conf = st.sidebar.slider('Ambang kepercayaan (confidence)', 0.05, 0.95, 0.25, 0.05)

files = st.file_uploader(
    'Unggah gambar (JPG/PNG)',
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
)

if files:
    if len(files) > MAKS_FOTO:
        st.warning(f'Maksimal {MAKS_FOTO} foto. Hanya {MAKS_FOTO} foto pertama yang diproses.')
        files = files[:MAKS_FOTO]

    for f in files:
        with st.expander(f.name, expanded=True):
            with st.spinner('Mendeteksi...'):
                asli, hasil_img, jumlah = deteksi(f.getvalue(), conf)

            k1, k2 = st.columns(2)
            k1.image(asli, caption='Foto asli', use_container_width=True)
            k2.image(hasil_img, caption='Hasil deteksi', use_container_width=True)

            if not jumlah:
                st.warning('Tidak ada motif terdeteksi. Coba turunkan ambang kepercayaan.')
            else:
                st.write('**Motif terdeteksi:** ' +
                         ', '.join(f'{n} ({j})' for n, j in sorted(jumlah.items(), key=lambda x: -x[1])))