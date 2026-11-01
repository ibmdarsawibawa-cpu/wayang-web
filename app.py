import io
from collections import Counter

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

MAKS_FOTO = 5
SISI_MAKS = 1024        # foto diperkecil dulu agar cepat & hemat RAM
UKURAN_MODEL = 640
CONF_DETEKSI = 0.05     # model dijalankan dengan ambang rendah; penyaringan dilakukan saat menampilkan
WARNA = ['#00C2FF', '#FF2D55', '#C8D400', '#FF9500', '#8E5CF7']

st.set_page_config(page_title='Deteksi Motif Ukiran Wayang Kulit Bali', layout='wide')
st.title('Deteksi Motif Ukiran Wayang Kulit Bali')
st.caption('Unggah foto atau ambil gambar dari kamera untuk mendeteksi motif ukiran wayang kulit.')


@st.cache_resource
def muat_model():
    return YOLO('best.pt')


model = muat_model()
NAMA_KELAS = [model.names[i] for i in sorted(model.names)]
WARNA_KELAS = {n: WARNA[i % len(WARNA)] for i, n in enumerate(NAMA_KELAS)}


@st.cache_data(show_spinner=False, max_entries=20)
def deteksi(data: bytes):
    gambar = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
    gambar.thumbnail((SISI_MAKS, SISI_MAKS))
    hasil = model.predict(gambar, conf=CONF_DETEKSI, imgsz=UKURAN_MODEL, verbose=False)[0]

    # simpan semua kandidat; hasil deteksi tetap sama, slider dan menu hanya menyaring tampilan
    xyxy = hasil.boxes.xyxy.cpu().numpy()
    kelas = hasil.boxes.cls.cpu().numpy().astype(int)
    skor = hasil.boxes.conf.cpu().numpy()
    daftar = [
        (model.names[int(c)], float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        for b, c, s in zip(xyxy, kelas, skor)
    ]
    return gambar, daftar


def muat_font(ukuran):
    try:
        return ImageFont.load_default(size=ukuran)
    except TypeError:
        return ImageFont.load_default()


def gambar_kotak(gambar, daftar):
    img = gambar.copy()
    draw = ImageDraw.Draw(img)
    tebal = max(2, img.width // 300)
    font = muat_font(max(12, img.width // 50))
    for nama, skor, x1, y1, x2, y2 in daftar:
        warna = WARNA_KELAS[nama]
        draw.rectangle([x1, y1, x2, y2], outline=warna, width=tebal)
        teks = f'{nama} {skor:.0%}'
        l, t, r, b = draw.textbbox((0, 0), teks, font=font)
        w, h = r - l + 6, b - t + 6
        y_teks = max(0, y1 - h)
        draw.rectangle([x1, y_teks, x1 + w, y_teks + h], fill=warna)
        draw.text((x1 + 3, y_teks + 1), teks, fill='white', font=font)
    return img


def tampilkan_hasil(nama, data, min_conf, tampil):
    with st.expander(nama, expanded=True):
        with st.spinner('Mendeteksi...'):
            asli, semua = deteksi(data)

        terpilih = [d for d in semua if d[0] in tampil and d[1] >= min_conf]
        jumlah = Counter(d[0] for d in terpilih)

        k1, k2 = st.columns(2)
        k1.image(asli, caption='Foto asli', use_container_width=True)
        k2.image(gambar_kotak(asli, terpilih), caption='Hasil deteksi', use_container_width=True)

        if not jumlah:
            st.warning('Tidak ada motif terdeteksi. Coba turunkan batas keyakinan '
                       'atau periksa menu "Tampilkan motif".')
        else:
            st.write('**Motif terdeteksi:** ' +
                     ', '.join(f'{n} ({j})' for n, j in sorted(jumlah.items(), key=lambda x: -x[1])))


min_conf = st.sidebar.slider(
    'Batas keyakinan yang ditampilkan', 0.05, 0.95, 0.20, 0.05,
    help='Hanya menyaring tampilan, tidak mengubah hasil deteksi model.',
)
tampil = st.sidebar.multiselect(
    'Tampilkan motif',
    NAMA_KELAS,
    default=NAMA_KELAS,
    help='Pilih satu atau lebih motif. Kotak motif lain disembunyikan.',
)
if not tampil:
    st.info('Pilih minimal satu motif di menu "Tampilkan motif" (sidebar kiri).')

# radio dipakai (bukan tab) supaya kamera hanya aktif kalau dipilih
sumber = st.radio('Sumber gambar', ['Unggah foto', 'Kamera webcam'], horizontal=True)

if sumber == 'Unggah foto':
    files = st.file_uploader(
        f'Unggah gambar (JPG/PNG), maksimal {MAKS_FOTO} foto',
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
    )
    if files:
        if len(files) > MAKS_FOTO:
            st.warning(f'Maksimal {MAKS_FOTO} foto. Hanya {MAKS_FOTO} foto pertama yang diproses.')
            files = files[:MAKS_FOTO]
        for f in files:
            tampilkan_hasil(f.name, f.getvalue(), min_conf, tampil)
else:
    foto = st.camera_input('Arahkan wayang ke kamera, lalu klik "Take Photo"')
    if foto is not None:
        tampilkan_hasil('Foto dari kamera', foto.getvalue(), min_conf, tampil)