import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime

# Page config
st.set_page_config(
    page_title="SkinCheck - Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS yang kompatibel dengan dark theme
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .header-description {
        font-size: 1rem;
        opacity: 0.8;
    }
    
    /* Card components */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: inherit;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Input sections */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
    
    /* Result containers */
    .result-benign {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
    }
    
    .result-malignant {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
    }
    
    /* Status boxes */
    .success-status {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-status {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-status {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-status {
        background: rgba(220, 53, 69, 0.1);
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ff6b6b;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #667eea;
        border-radius: 10px;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Camera input */
    .stCameraInput > div {
        border: 2px solid #667eea;
        border-radius: 10px;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Footer */
    .footer-container {
        background: rgba(44, 62, 80, 0.8);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("skincheck_model_tf.keras")
        return model, None
    except Exception as e:
        return None, str(e)

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🔬 SkinCheck</div>
    <div class="header-subtitle">AI-Powered Skin Disease Detection</div>
    <div class="header-description">Deteksi cepat dan akurat untuk klasifikasi penyakit kulit menggunakan Computer Vision</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 About SkinCheck")
    
    st.markdown("""
    *SkinCheck* menggunakan teknologi Computer Vision untuk membantu klasifikasi awal penyakit kulit dengan akurasi tinggi.
    """)
    
    st.markdown("### 🚀 Features:")
    st.markdown("""
    • 📤 Image upload atau camera input  
    • 🤖 Real-time analysis  
    • 📊 Confidence-based results  
    • 👨‍⚕ User-friendly interface  
    """)
    
    st.markdown("---")
    
    st.markdown("### 📖 Cara Penggunaan:")
    st.markdown("""
    *Step 1:* Pilih metode input gambar  
    *Step 2:* Upload foto atau gunakan kamera  
    *Step 3:* Klik tombol 'Mulai Analisis'  
    *Step 4:* Lihat hasil dan rekomendasi  
    """)
    
    st.markdown("---")
    
    st.warning("⚠ Medical Disclaimer:** Hasil ini hanya untuk skrining awal. Selalu konsultasikan dengan dokter spesialis kulit untuk diagnosis dan penanganan yang tepat.")
    
    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    st.info("""
    *Model:* TensorFlow Keras  
    *Classes:* Benign, Malignant  
    *Input Size:* 224x224 pixels  
    *Accuracy:* ~95% on test data
    """)

# Main content
st.markdown('<div class="section-header">📤 Upload & Analyze</div>', unsafe_allow_html=True)

# Create columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🖼 Input Gambar")
    
    # Input method selection
    input_mode = st.radio(
        "*Pilih Metode Input:*",
        ["📁 Upload Gambar", "📸 Gunakan Kamera"],
        help="Pilih metode untuk memasukkan gambar"
    )
    
    img = None
    
    if input_mode == "📁 Upload Gambar":
        st.markdown('<div class="upload-area">📁 Drag & Drop atau Browse untuk Upload</div>', unsafe_allow_html=True)
        
        st.info("""
        *💡 Tips Upload:*  
        • Gunakan foto dengan pencahayaan yang baik  
        • Pastikan area kulit terlihat jelas dan fokus  
        • Format: JPG, JPEG, PNG (max 5MB)  
        """)
        
        uploaded_file = st.file_uploader(
            "Pilih gambar kulit",
            type=["jpg", "jpeg", "png"],
            help="Upload gambar dengan format JPG, JPEG, atau PNG"
        )
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.success("✅ *Gambar berhasil diupload!* Siap untuk dianalisis.")
            
    elif input_mode == "📸 Gunakan Kamera":
        st.markdown('<div class="upload-area">📸 Ambil Foto dengan Kamera</div>', unsafe_allow_html=True)
        
        st.info("""
        *💡 Tips Kamera:*  
        • Pastikan pencahayaan cukup terang  
        • Fokuskan pada area kulit yang ingin diperiksa  
        • Hindari bayangan dan pantulan cahaya  
        """)
        
        camera_input = st.camera_input("📷 Ambil foto dengan kamera")
        
        if camera_input is not None:
            img = Image.open(camera_input)
            st.success("✅ *Foto berhasil diambil!* Siap untuk dianalisis.")

with col2:
    st.markdown("### 🔍 Preview & Hasil Analisis")
    
    if img is not None:
        # Display image
        st.markdown("#### 🖼 Gambar Input:")
        st.image(img, caption="Gambar yang diinput", use_container_width=True)
        
        # Load model
        model, error = load_model()
        
        if model is not None:
            # Analysis button
            if st.button("🔬 *MULAI ANALISIS*", key="analyze_btn"):
                with st.spinner("🔄 Menganalisis gambar..."):
                    try:
                        # Preprocess image
                        img_resized = img.resize((224, 224))
                        img_array = tf.keras.utils.img_to_array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0) / 255.0
                        
                        # Prediction
                        pred = model.predict(img_array, verbose=0)
                        labels = ['Benign (Jinak)', 'Malignant (Ganas)']
                        kelas = np.argmax(pred)
                        confidence = float(pred[0][kelas])
                        
                        # Results
                        st.markdown("---")
                        st.markdown("#### 📊 Hasil Analisis")
                        
                        # Display result
                        result_class = "result-benign" if kelas == 0 else "result-malignant"
                        result_icon = "🟢" if kelas == 0 else "🔴"
                        
                        st.markdown(f"""
                        <div class="{result_class}">
                            <h3>{result_icon} Hasil Prediksi</h3>
                            <h2>{labels[kelas]}</h2>
                            <p style="font-size: 1.2rem; margin-top: 1rem;">
                                <strong>Confidence Score: {confidence:.1%}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.markdown("#### 📈 Confidence Level")
                        st.progress(confidence)
                        
                        # Confidence interpretation
                        if confidence >= 0.85:
                            st.success(f"🟢 *Tingkat Keyakinan: Sangat Tinggi* ({confidence:.1%})")
                        elif confidence >= 0.70:
                            st.info(f"🔵 *Tingkat Keyakinan: Tinggi* ({confidence:.1%})")
                        elif confidence >= 0.55:
                            st.warning(f"🟡 *Tingkat Keyakinan: Sedang* ({confidence:.1%})")
                        else:
                            st.error(f"🔴 *Tingkat Keyakinan: Rendah* ({confidence:.1%})")
                        
                        # Medical recommendations
                        st.markdown("#### 🏥 Rekomendasi Medis")
                        
                        if kelas == 0:  # Benign
                            st.markdown("""
                            <div class="success-status">
                                <strong>✅ Hasil: Kemungkinan Jinak (Benign)</strong><br><br>
                                Berdasarkan analisis, kondisi kulit kemungkinan bersifat jinak. 
                                Namun, tetap disarankan untuk:<br>
                                • Lakukan pemeriksaan rutin setiap 6 bulan<br>
                                • Perhatikan perubahan bentuk, warna, atau ukuran<br>
                                • Konsultasi dengan dermatologist jika ada keraguan
                            </div>
                            """, unsafe_allow_html=True)
                        else:  # Malignant
                            st.markdown("""
                            <div class="warning-status">
                                <strong>⚠ Hasil: Kemungkinan Ganas (Malignant)</strong><br><br>
                                <strong>PENTING - SEGERA AMBIL TINDAKAN:</strong><br>
                                • 🚨 Konsultasi dengan dokter spesialis kulit SEGERA<br>
                                • 📅 Buat janji dengan dermatologist dalam 1-2 hari<br>
                                • 📋 Bawa hasil ini untuk referensi dokter<br>
                                • 🔍 Mungkin diperlukan biopsi untuk konfirmasi
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Additional info
                        st.markdown("#### 📝 Detail Analisis")
                        
                        # Prediction probabilities
                        st.markdown("*Probabilitas untuk setiap kelas:*")
                        for i, label in enumerate(labels):
                            prob = float(pred[0][i])
                            st.write(f"• {label}: {prob:.1%}")
                        
                        # Timestamp
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.caption(f"Analisis dilakukan pada: {now}")
                        
                    except Exception as e:
                        st.error(f"❌ *Error dalam Analisis:* {str(e)}")
                        st.info("Silakan coba lagi atau gunakan gambar yang berbeda.")
        else:
            st.error(f"❌ *Model Tidak Dapat Dimuat*  \nError: {error}")
    else:
        # Welcome card with better styling
        st.markdown("""
        <div class="feature-card">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">👈 Mulai dengan Upload Gambar</h3>
                <p style="opacity: 0.8; font-size: 1.1rem;">Pilih gambar kulit menggunakan salah satu metode di sebelah kiri untuk memulai analisis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis features
        st.markdown("#### 🔍 *Yang Akan Dianalisis:*")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            🧬 *Morfologi Lesi*  
            • Bentuk dan batas lesi  
            • Simetri struktur kulit  
            
            🎨 *Analisis Visual*  
            • Pola warna dominan  
            • Variasi tekstur permukaan  
            """)
        
        with col_b:
            st.markdown("""
            🔬 *Karakteristik Klinis*  
            • Indikator visual mencurigakan  
            • Pattern recognition Neural Network 
            
            📊 *Hasil Klasifikasi*  
            • Probabilitas jinak vs ganas  
            • Confidence score analysis  
            """)
        
        # Analysis time info
        st.markdown("---")
        st.markdown("#### ⏱ *Informasi Analisis:*")
        
        col_time1, col_time2, col_time3 = st.columns(3)
        
        with col_time1:
            st.metric(
                label="⚡ Waktu Proses",
                value="2-3 detik",
                help="Waktu rata-rata untuk analisis lengkap"
            )
        
        with col_time2:
            st.metric(
                label="🎯 Akurasi Model",
                value="~95%",
                help="Akurasi pada dataset testing"
            )
        
        with col_time3:
            st.metric(
                label="📸 Format Support",
                value="JPG, PNG",
                help="Format gambar yang didukung"
            )

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-container">
    <h3>⚠ Medical Disclaimer</h3>
    <p>
        <strong>SkinCheck</strong> adalah alat bantu skrining awal dan tidak menggantikan 
        konsultasi medis profesional. Aplikasi ini menggunakan Neural Network untuk memberikan 
        estimasi berdasarkan pola visual, namun diagnosis resmi harus dilakukan 
        oleh dokter spesialis kulit yang berkualifikasi.
    </p>
    <p style="margin-top: 1rem;">
        Selalu konsultasikan kondisi kulit Anda dengan dermatologist untuk 
        diagnosis akurat dan penanganan yang tepat.
    </p>
    <hr style="margin: 2rem 0; opacity: 0.3;">
    <p style="font-size: 0.9rem; opacity: 0.8;">
        <em>SkinCheck v2.1 - Powered by TensorFlow & Streamlit</em><br>
        Made with ❤ for better healthcare accessibility
    </p>
</div>
""", unsafe_allow_html=True)