import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Setup Streamlit
st.set_page_config(page_title="Visualisasi EEG & BPM", layout="wide")
st.title("🧠 Visualisasi EEG dan BPM Mahasiswa")

# Fungsi bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Kolom kiri-kanan untuk upload file
col1, col2 = st.columns(2)

with col1:
    eeg_file = st.file_uploader("📁 EEG CSV", type="csv")

with col2:
    bpm_file = st.file_uploader("📁 BPM CSV", type="csv")

# Sampling rate
fs = 512

if eeg_file is not None and bpm_file is not None:
    eeg_data = pd.read_csv(eeg_file)
    bpm_data = pd.read_csv(bpm_file)

    # Imputasi nilai kosong
    eeg_data.fillna(eeg_data.select_dtypes(include=[np.number]).mean(), inplace=True)
    bpm_data.fillna(bpm_data.select_dtypes(include=[np.number]).mean(), inplace=True)

    with col1:
        fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(6, 9), dpi=100)
        colors = {'Low Alpha': 'blue', 'High Alpha': 'green', 'Low Beta': 'purple', 'High Beta': 'black'}

        for i, (band, lowcut, highcut) in enumerate([('Low Alpha', 8, 10), ('High Alpha', 10, 12), ('Low Beta', 12, 18), ('High Beta', 18, 30)]):
            if band in eeg_data.columns:
                eeg_data[f'{band} Filtered'] = bandpass_filter(eeg_data[band], lowcut, highcut, fs)
                axs[i].plot(eeg_data['Timestamp'], eeg_data[f'{band} Filtered'], color=colors[band])
                axs[i].set_title(f'{band} Filtered', fontweight='bold')  # Make title bold
                axs[i].set_xlabel('Timestamp', fontweight='bold')
                axs[i].set_ylabel('Amplitude', fontweight='bold')
                axs[i].grid(True)
                axs[i].set_facecolor('yellow')
                axs[i].xaxis.set_major_locator(plt.MaxNLocator(5))
                axs[i].xaxis.set_ticklabels([])  # Hide the x-axis labels

        # Plot BPM time-series
        axs[-1].plot(bpm_data['Timestamp'], bpm_data['Avg BPM'], color='red')
        axs[-1].set_title('Avg BPM Time-Series', fontweight='bold')  # Make title bold
        axs[-1].set_xlabel('Timestamp', fontweight='bold')
        axs[-1].set_ylabel('Avg BPM', fontweight='bold')
        axs[-1].grid(True)
        axs[-1].set_facecolor('yellow')
        axs[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[-1].xaxis.set_ticklabels([])  # Hide the x-axis labels

        fig.tight_layout()
        st.pyplot(fig)

        if st.button("💾 Simpan Gambar", key="save_plot"):
            fig.savefig("hasil_visualisasi.png", dpi=89.6)
            st.success("✅ Gambar disimpan sebagai hasil_visualisasi.png")

    with col2:
        st.markdown("### 🔮 Hasil Prediksi Model")
        st.info("Model H5 bisa ditambahkan di sini nanti.")
