import numpy as np
import librosa
from scipy.signal import butter, filtfilt

TARGET_SR = 8000
DURATION = 5

def bandpass_filter(signal, sr=TARGET_SR, lowcut=50.0, highcut=2500.0, order=4):
    nyquist = 0.5 * sr
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)

def process_audio(file_path):
    y, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    y_clean = bandpass_filter(y)

    target_samples = TARGET_SR * DURATION
    if len(y_clean) < target_samples:
        y_clean = np.pad(y_clean, (0, target_samples - len(y_clean)))
    else:
        y_clean = y_clean[:target_samples]

    mel = librosa.feature.melspectrogram(y=y_clean, sr=TARGET_SR, n_mels=64, fmax=2500)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return y_clean, log_mel.astype(np.float32)