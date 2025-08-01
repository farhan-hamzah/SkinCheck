# 🔬 SkinCheck - Deteksi Penyakit Kulit dengan AI

**SkinCheck** adalah aplikasi berbasis web yang memanfaatkan teknologi **Computer Vision** dan **Deep Learning** untuk melakukan deteksi dini terhadap penyakit kulit dari citra gambar. Proyek ini dibuat dalam rangka lomba *Digiwar* dengan fokus pada pemanfaatan AI untuk meningkatkan akses kesehatan.


## 🚀 Fitur Utama

- 📷 Input gambar dari kamera atau upload file
- 🔍 Deteksi otomatis jenis penyakit kulit (jinak atau ganas)
- 📊 Confidence score & interpretasi hasil
- 💡 UI/UX ramah pengguna berbasis **Streamlit**
- 🧠 Model deep learning berbasis **ResNet50**

---

## 🧠 Teknologi yang Digunakan

| Komponen      | Teknologi                                   |
|---------------|----------------------------------------------|
| Frontend      | [Streamlit](https://streamlit.io/)          |
| Model AI      | TensorFlow + Keras, arsitektur ResNet50     |
| Gambar Input  | Pillow (PIL), OpenCV                        |
| Deployment    | Hugging Face Spaces (Non-Docker + app.toml) |
| Bahasa        | Python                                      |

---

## 📁 Struktur Folder

```
SkinCheck/
├── .streamlit/
│   └── config.toml          # Konfigurasi port Streamlit
├── dataSetDigistar/
│   ├── test/
│   │   ├── benign/          # Dataset test - tumor jinak
│   │   └── malignant/       # Dataset test - tumor ganas
│   └── train/
│       ├── benign/          # Dataset train - tumor jinak
│       └── malignant/       # Dataset train - tumor ganas
├── mesin.py                 # Kode untuk load dataset
├── save.py                  # Kode untuk menyimpan file .keras di lokal
├── app.py                   # Program utama (streamlit run app.py)
├── skincheck_model_tf.keras # Model hasil pelatihan
├── requirements.txt         # Dependensi aplikasi
└── README.md               # Dokumentasi proyek
```

---

## ⚙️ Cara Menjalankan di Lokal

### 1. Clone repositori
```bash
git clone https://github.com/farhan-hamzah/SkinCheck
cd SkinCheck
```

### 2. Buat virtual environment (Opsional tapi disarankan)
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi streamlit
```bash
streamlit run app.py
```

---

## 🧪 Cara Kerja Model

### 1. Preprocessing Gambar
Gambar diubah ke ukuran 224x224, distandarisasi, dan dinormalisasi.

### 2. Prediksi oleh Model
Model ResNet50 melakukan klasifikasi ke dalam 2 kelas:
- **Benign** (Jinak)
- **Malignant** (Ganas)

### 3. Output
Aplikasi memberikan hasil klasifikasi serta confidence score dan interpretasi risiko.

---

## 📈 Dataset

Model dilatih menggunakan dataset dari Kaggle:
**Skin Cancer MNIST: HAM10000**

---

## 🌍 Dampak & Manfaat

- 📱 Akses mudah untuk deteksi awal penyakit kulit
- 🩺 Bermanfaat sebagai alat bantu skrining cepat
- 🧑‍⚕️ Tidak menggantikan diagnosis dokter, tetapi memberi insight awal
- 🔭 Potensi pengembangan untuk klasifikasi lebih banyak jenis penyakit kulit

---

## 🔐 Disclaimer Medis

⚠️ **Penting untuk diingat:**
- SkinCheck tidak menggantikan peran dokter
- Hasil dari aplikasi ini hanya sebagai skrining awal
- Untuk diagnosis dan penanganan lebih lanjut, konsultasikan dengan dokter kulit profesional

---

## 👨‍💻 Author

**Farhan Hamzah**
- Mahasiswa Informatika Universitas Telkom 
- Memiliki ketertarikan dengan Artificial Intelligence dan Computer Vision
- 🔗 GitHub: [@farhan-hamzah](https://github.com/farhan-hamzah)

---

## 📄 Lisensi

MIT License © 2025 Farhan Hamzah

Bebas digunakan untuk pembelajaran dan pengembangan lanjutan.