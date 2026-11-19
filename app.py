import streamlit as st
import importlib

try:
    np = importlib.import_module("numpy")
except ImportError:
    st.error("NumPy is required. Install it with: pip install numpy")
    st.stop()

import matplotlib.pyplot as plt
import librosa.display

try:
    ort = importlib.import_module("onnxruntime")
except ImportError:
    ort = None

from preprocess import process_audio, TARGET_SR

CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]

st.set_page_config(page_title="Deep Listening", layout="wide")
st.title("Deep Listening: CNN Acoustic Lung Sound Screening")
st.caption("Non-invasive early lung disease screening pipeline | Triage Prototype")

@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession("lung_cnn.onnx")

session = load_onnx_model()
input_name = session.get_inputs()[0].name

uploaded_file = st.file_uploader("Upload Chest Sound Recording (.wav)", type=["wav"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio("temp_audio.wav")

    y_clean, log_mel = process_audio("temp_audio.wav")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Filtered Waveform (50–2500 Hz)")
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(y_clean, sr=TARGET_SR, ax=ax1)
        ax1.set_xlabel("Time (s)")
        st.pyplot(fig1)

    with col2:
        st.subheader("Log-Mel Spectrogram")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        img = librosa.display.specshow(log_mel, sr=TARGET_SR, x_axis="time", y_axis="mel", fmax=2500, ax=ax2)
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2)

    # Run fast ONNX inference
    model_input = log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)
    probs = session.run(None, {input_name: model_input})[0][0]

    top_idx = int(np.argmax(probs))
    top_class = CLASSES[top_idx]
    confidence = probs[top_idx] * 100

    st.divider()
    res1, res2 = st.columns([1, 2])
    with res1:
        st.metric("Flagged Acoustic Event", top_class, f"{confidence:.1f}% Confidence")
        report = (
            f"DEEP LISTENING SCREENING REPORT\n"
            f"Primary Flag: {top_class} ({confidence:.2f}%)\n\n"
            + "\n".join([f"{c}: {p*100:.2f}%" for c, p in zip(CLASSES, probs)])
            + "\n\nDisclaimer: Triage prototype only; requires clinical verification."
        )
        st.download_button("Download Triage Report", data=report, file_name="screening_report.txt")

    with res2:
        st.subheader("Acoustic Event Probability Breakdown")
        st.bar_chart({CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))})