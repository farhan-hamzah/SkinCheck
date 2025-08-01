# 🔬 SkinCheck - Deteksi Penyakit Kulit dengan AI

**SkinCheck** adalah aplikasi berbasis web yang memanfaatkan teknologi **Computer Vision** dan **Deep Learning** untuk melakukan deteksi dini terhadap penyakit kulit dari citra gambar. Proyek ini dibuat dalam rangka lomba *Digiwar* dengan fokus pada pemanfaatan AI untuk meningkatkan akses kesehatan.

![SkinCheck Banner](https://your-image-link-if-any.com)

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
## 🗂️ Struktur Folder

```plaintext
SkinCheck/
├── .streamlit/
│   └── app.toml               # Konfigurasi port Streamlit
├── dataSetDigistar/
│   ├── test/
│   │   ├── benign/
│   │   └── malignant/
│   └── train/
│       ├── benign/
│       └── malignant/
├── mesin.py                   # Kode untuk load dataset
├── save.py                    # Kode untuk menyimpan file .keras di lokal
├── app.py                     # Program utama (streamlit run app.py)
├── skincheck_model_tf.keras  # Model hasil pelatihan
├── requirements.txt           # Dependensi aplikasi
└── README.md                  # Dokumentasi proyek



---

## ⚙️ Cara Menjalankan di Lokal

1. **Clone repositori**  
   ```bash
   git clone https://github.com/farhan-hamzah/SkinCheck
   cd SkinCheck

2. **Buat virtual enviroment (Opsional tadi disarankan)**
python -m venv venv
source venv/bin/activate  # di Linux/Mac
.\venv\Scripts\activate    # di Windows

3. **Install depedensi**
pip install -r requirements.txt

4. **Jalankan aplikasi streamlit**
streamlit run app.py

🧪 Cara Kerja Model
1. **Preprocessing Gambar:**
Gambar diubah ke ukuran 224x224, distandarisasi, dan dinormalisasi.

2. **Prediksi oleh Model:**
Model ResNet50 melakukan klasifikasi ke dalam 2 kelas:
    -Benign (Jinak)
    -Malignant (Ganas)

3. **Output:**
Aplikasi memberikan hasil klasifikasi serta confidence score dan interpretasi risiko.

📈 Dataset
Model dilatih menggunakan dataset dari Kaggle:
Skin Cancer MNIST: HAM10000

🌍 Dampak & Manfaat
📱 Akses mudah untuk deteksi awal penyakit kulit
🩺 Bermanfaat sebagai alat bantu skrining cepat
🧑‍⚕️ Tidak menggantikan diagnosis dokter, tetapi memberi insight awal
🔭 Potensi pengembangan untuk klasifikasi lebih banyak jenis penyakit kulit

🔐 Disclaimer Medis
SkinCheck tidak menggantikan peran dokter.
Hasil dari aplikasi ini hanya sebagai skrining awal.
Untuk diagnosis dan penanganan lebih lanjut, konsultasikan dengan dokter kulit profesional.

📌 Author
Farhan Hamzah
👨‍💻 Mahasiswa Informatika Universitas Telkom Yang Memiliki Ketertarikan Dengan Artificial Intelligence dan Computer Vision
🔗 GitHub: @farhan-hamzah

📄 Lisensi
MIT License © 2025 Farhan Hamzah
Bebas digunakan untuk pembelajaran dan pengembangan lanjutan.

