import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path

import paho.mqtt.publish as publish
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

SISI_MAKS = 1024        # foto diperkecil dulu agar cepat & hemat RAM
UKURAN_MODEL = 640
CONF_DETEKSI = 0.05     # model dijalankan dengan ambang rendah; penyaringan dilakukan saat menampilkan
WARNA = ['#00C2FF', '#FF2D55', '#C8D400', '#FF9500', '#8E5CF7']

# Pengiriman hasil ke ESP32 (harus SAMA PERSIS dengan TOPIC di kode ESP32)
MQTT_HOST = 'broker.hivemq.com'
MQTT_TOPIC = 'deteksi-ukiran-wayang-kulit-bali-x7k2/hasil'

# Tombol di alat (harus SAMA PERSIS dengan TOPIC_KONTROL di kode ESP32)
MQTT_WS = 'wss://broker.hivemq.com:8884/mqtt'
TOPIK_KONTROL = 'deteksi-ukiran-wayang-kulit-bali-x7k2/kontrol'
DETIK_HITUNG = 3        # hitung mundur sebelum foto diambil (isi 3 sampai 5)

# JavaScript yang berjalan di browser: mendengarkan tombol di alat
JS_TOMBOL = """
<script src="https://cdn.jsdelivr.net/npm/mqtt@5.10.1/dist/mqtt.min.js"></script>
<script>
const DETIK = __DETIK__;
const pdoc = () => window.parent.document;
const tunggu = (ms) => new Promise((r) => setTimeout(r, ms));
let sibuk = false;

function cariTombol(teks) {
  const semua = pdoc().querySelectorAll('button');
  for (const b of semua) {
    if ((b.textContent || '').toLowerCase().includes(teks)) return b;
  }
  return null;
}
const tombolAmbil = () => cariTombol('take photo');
const tombolHapus = () => cariTombol('clear photo');

function bunyi(frek, ms) {
  try {
    const a = new (window.AudioContext || window.webkitAudioContext)();
    const osc = a.createOscillator();
    osc.frequency.value = frek;
    osc.connect(a.destination);
    osc.start();
    setTimeout(() => { osc.stop(); a.close(); }, ms);
  } catch (e) {}
}

function buatOverlay() {
  let o = pdoc().getElementById('hitung-mundur');
  if (!o) {
    o = pdoc().createElement('div');
    o.id = 'hitung-mundur';
    o.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:999999;' +
      'display:flex;align-items:center;justify-content:center;' +
      'background:rgba(0,0,0,0.45);color:#fff;font-size:40vh;font-weight:700;' +
      'font-family:sans-serif;pointer-events:none;';
    pdoc().body.appendChild(o);
  }
  return o;
}

// pastikan mode "Kamera webcam" aktif (pindahkan otomatis kalau belum)
async function pastikanKamera() {
  if (tombolAmbil() || tombolHapus()) return true;
  const labels = pdoc().querySelectorAll('label');
  for (const l of labels) {
    if ((l.textContent || '').includes('Kamera webcam')) { l.click(); break; }
  }
  for (let i = 0; i < 40; i++) {
    await tunggu(200);
    if (tombolAmbil() || tombolHapus()) return true;
  }
  return false;
}

async function ambilFoto() {
  if (sibuk) return;
  sibuk = true;
  try {
    if (!(await pastikanKamera())) {
      console.log('foto: kamera tidak ditemukan');
      return;
    }
    const hapus = tombolHapus();
    if (hapus) { hapus.click(); await tunggu(1500); }   // siapkan kamera lagi
    const o = buatOverlay();
    for (let n = DETIK; n > 0; n--) {
      o.textContent = n;
      bunyi(660, 150);
      await tunggu(1000);
    }
    o.textContent = '\\u{1F4F8}';
    bunyi(990, 300);
    const ambil = tombolAmbil();
    if (ambil) { ambil.click(); }
    else { console.log('foto: tombol Take Photo tidak ditemukan'); }
    await tunggu(700);
    o.remove();
  } finally {
    sibuk = false;
  }
}

const client = mqtt.connect('__WS__');
client.on('connect', () => {
  console.log('MQTT terhubung');
  client.subscribe('__TOPIK__');
});
client.on('message', (topic, msg) => {
  const aksi = msg.toString();
  console.log('pesan tombol:', aksi);
  if (aksi === 'reset') {
    try { window.parent.location.reload(); }
    catch (e) { window.top.location.reload(); }
  } else if (aksi === 'foto') {
    ambilFoto();
  }
});
</script>
"""


def dengarkan_tombol():
    """Dengarkan tombol di alat (lewat MQTT) dan jalankan aksinya di halaman."""
    html = (JS_TOMBOL
            .replace('__WS__', MQTT_WS)
            .replace('__TOPIK__', TOPIK_KONTROL)
            .replace('__DETIK__', str(DETIK_HITUNG)))
    components.html(html, height=0)


def muat_info():
    """Baca penjelasan tiap motif dari info_ukiran.json (satu sumber untuk web dan suara)."""
    try:
        with open(Path(__file__).parent / 'info_ukiran.json', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return {}


INFO_UKIRAN = muat_info()

st.set_page_config(page_title='Deteksi Motif Ukiran Wayang Kulit Bali', layout='wide')
st.title('Deteksi Motif Ukiran Wayang Kulit Bali')
st.caption('Unggah foto atau ambil gambar dari kamera untuk mendeteksi motif ukiran wayang kulit.')
dengarkan_tombol()


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


def kirim_ke_alat(daftar_kelas):
    """Kirim nomor kelas yang terdeteksi ke ESP32 lewat broker MQTT."""
    try:
        pesan = json.dumps({'id': int(time.time()), 'kelas': daftar_kelas})
        publish.single(MQTT_TOPIC, pesan, hostname=MQTT_HOST, port=1883, keepalive=10)
        return True
    except Exception:
        return False


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

            # --- penjelasan tiap motif (teks sama dengan yang dibacakan speaker) ---
            st.markdown('#### Penjelasan motif')
            for n, j in sorted(jumlah.items(), key=lambda x: -x[1]):
                info = INFO_UKIRAN.get(n, {})
                judul = info.get('nama', n)
                isi = info.get('penjelasan', 'Penjelasan belum tersedia.')
                st.markdown(f'**{judul}** ({j} terdeteksi)\n\n> {isi}')

        # --- kirim hasil ke ESP32 (speaker) ---
        daftar_kelas = sorted({NAMA_KELAS.index(d[0]) for d in terpilih})
        tanda = hashlib.md5(data).hexdigest() + str(daftar_kelas)
        terkirim = st.session_state.setdefault('terkirim', set())
        if kirim_otomatis and tanda not in terkirim:
            if kirim_ke_alat(daftar_kelas):
                terkirim.add(tanda)
                st.success('Hasil dikirim ke speaker.')
            else:
                st.error('Gagal mengirim ke alat (cek internet / broker).')
        if st.button('🔊 Bunyikan lagi di alat', key='btn_' + tanda):
            if kirim_ke_alat(daftar_kelas):
                st.success('Dikirim ulang ke speaker.')
            else:
                st.error('Gagal mengirim ke alat.')


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
kirim_otomatis = st.sidebar.checkbox('Kirim hasil ke speaker (ESP32)', value=True)
if not tampil:
    st.info('Pilih minimal satu motif di menu "Tampilkan motif" (sidebar kiri).')

# radio dipakai (bukan tab) supaya kamera hanya aktif kalau dipilih
# index=1 -> mode kamera aktif sejak awal, supaya tombol Foto di alat langsung bisa dipakai
sumber = st.radio('Sumber gambar', ['Unggah foto', 'Kamera webcam'], index=1, horizontal=True)

if sumber == 'Unggah foto':
    f = st.file_uploader(
        'Unggah satu gambar (JPG/PNG)',
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False,
    )
    if f is not None:
        tampilkan_hasil(f.name, f.getvalue(), min_conf, tampil)
else:
    foto = st.camera_input('Arahkan wayang ke kamera, lalu klik "Take Photo"')
    if foto is not None:
        tampilkan_hasil('Foto dari kamera', foto.getvalue(), min_conf, tampil)
