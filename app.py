import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import io

# Setup Streamlit
st.set_page_config(page_title="Visualisasi EEG & BPM", layout="wide")
st.title("🧠 FocusNet: Identifikasi Tingkat Fokus berbasis Multimodal")

# Fungsi bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=1):
    """
    Bandpass filter untuk EEG signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Validasi parameter
    if low <= 0 or high >= 1:
        st.error("⚠️ Frekuensi filter tidak valid untuk sampling rate ini")
        return data
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Sampling rate
fs = 512

# Kolom kiri-kanan untuk upload file
col1, col2 = st.columns(2)

with col1:
    eeg_file = st.file_uploader("📁 Upload EEG CSV", type="csv", key="eeg_upload")

with col2:
    bpm_file = st.file_uploader("📁 Upload BPM CSV", type="csv", key="bpm_upload")

# Main logic
if eeg_file is not None and bpm_file is not None:
    try:
        # Load data
        eeg_data = pd.read_csv(eeg_file)
        bpm_data = pd.read_csv(bpm_file)
        
        # Validasi data
        if eeg_data.empty or bpm_data.empty:
            st.error("❌ File CSV kosong atau tidak valid")
            st.stop()
        
        # Imputasi nilai kosong
        numeric_eeg = eeg_data.select_dtypes(include=[np.number])
        numeric_bpm = bpm_data.select_dtypes(include=[np.number])
        
        if not numeric_eeg.empty:
            eeg_data.fillna(numeric_eeg.mean(), inplace=True)
        if not numeric_bpm.empty:
            bpm_data.fillna(numeric_bpm.mean(), inplace=True)
        
        # Kolom untuk visualisasi
        with col1:
            st.subheader("📊 EEG & BPM Time-Series")
            
            # Create figure dengan subplots
            fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(10, 12), dpi=100)
            fig.patch.set_facecolor('white')
            
            # Define bands dengan warna
            bands = [
                ('Low Alpha', 8, 10, 'blue'),
                ('High Alpha', 10, 12, 'green'),
                ('Low Beta', 12, 18, 'purple'),
                ('High Beta', 18, 30, 'darkblue')
            ]
            
            # Plot EEG bands
            for i, (band, lowcut, highcut, color) in enumerate(bands):
                if band in eeg_data.columns:
                    # Apply bandpass filter
                    eeg_data[f'{band}_filtered'] = bandpass_filter(
                        eeg_data[band].values, lowcut, highcut, fs
                    )
                    
                    # Plot
                    axs[i].plot(
                        range(len(eeg_data)), 
                        eeg_data[f'{band}_filtered'], 
                        color=color, 
                        linewidth=1.5
                    )
                    axs[i].set_title(f'{band} Filtered (Bandpass {lowcut}-{highcut} Hz)', 
                                    fontweight='bold', fontsize=11)
                    axs[i].set_ylabel('Amplitude (µV)', fontweight='bold')
                    axs[i].grid(True, alpha=0.3)
                    axs[i].set_facecolor('#f0f0f0')
                else:
                    st.warning(f"⚠️ Kolom '{band}' tidak ditemukan di file EEG")
            
            # Plot BPM time-series
            if 'Avg BPM' in bpm_data.columns:
                axs[-1].plot(
                    range(len(bpm_data)), 
                    bpm_data['Avg BPM'], 
                    color='red', 
                    linewidth=2
                )
                axs[-1].set_title('Avg BPM Time-Series', fontweight='bold', fontsize=11)
                axs[-1].set_ylabel('BPM', fontweight='bold')
                axs[-1].set_xlabel('Sample Index', fontweight='bold')
                axs[-1].grid(True, alpha=0.3)
                axs[-1].set_facecolor('#f0f0f0')
            else:
                st.warning("⚠️ Kolom 'Avg BPM' tidak ditemukan di file BPM")
            
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # Download button untuk menyimpan gambar
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="💾 Download Gambar (PNG)",
                data=buf,
                file_name="hasil_visualisasi_eeg_bpm.png",
                mime="image/png"
            )
        
        # Kolom kanan untuk statistik dan prediksi
        with col2:
            st.subheader("📈 Statistik & Analisis")
            
            # Statistik EEG
            st.markdown("#### EEG Statistics")
            eeg_numeric = eeg_data.select_dtypes(include=[np.number])
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.metric("EEG Mean", f"{eeg_numeric.mean().mean():.2f}")
                st.metric("EEG Std Dev", f"{eeg_numeric.std().mean():.2f}")
            
            with col_stat2:
                st.metric("EEG Min", f"{eeg_numeric.min().min():.2f}")
                st.metric("EEG Max", f"{eeg_numeric.max().max():.2f}")
            
            # Statistik BPM
            st.markdown("#### BPM Statistics")
            if 'Avg BPM' in bpm_data.columns:
                col_bpm1, col_bpm2 = st.columns(2)
                
                with col_bpm1:
                    st.metric("Average BPM", f"{bpm_data['Avg BPM'].mean():.1f}")
                    st.metric("Min BPM", f"{bpm_data['Avg BPM'].min():.1f}")
                
                with col_bpm2:
                    st.metric("Max BPM", f"{bpm_data['Avg BPM'].max():.1f}")
                    st.metric("Std Dev BPM", f"{bpm_data['Avg BPM'].std():.1f}")
            
            # Info untuk model prediksi
            st.markdown("#### 🔮 Model Prediction")
            st.info(
                "✨ **Ruang untuk Model H5**\n\n"
                "Untuk menambahkan model H5:\n"
                "1. Simpan model TensorFlow/Keras sebagai `.h5`\n"
                "2. Load dengan `keras.models.load_model()`\n"
                "3. Gunakan EEG features untuk prediksi"
            )
    
    except pd.errors.EmptyDataError:
        st.error("❌ File CSV kosong atau format tidak valid")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.write("Pastikan file CSV memiliki format yang benar dengan kolom yang diperlukan")

else:
    st.info("👆 Silakan upload kedua file CSV (EEG dan BPM) untuk memulai visualisasi")

