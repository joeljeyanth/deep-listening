import subprocess
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
import imageio_ffmpeg

TARGET_SR = 8000
DURATION = 3
TARGET_SAMPLES = TARGET_SR * DURATION

def bandpass_filter(signal, sr=TARGET_SR, lowcut=50.0, highcut=2500.0, order=4):
    nyquist = 0.5 * sr
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)

def convert_to_wav(input_path, wav_path="converted_audio.wav"):
    if input_path.lower().endswith(".wav"):
        return input_path
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", str(TARGET_SR), "-ac", "1", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return wav_path

def make_window_spectrogram(segment):
    if np.max(np.abs(segment)) > 0:
        segment = segment / np.max(np.abs(segment))
    if len(segment) < TARGET_SAMPLES:
        segment = np.pad(segment, (0, TARGET_SAMPLES - len(segment)))
    else:
        segment = segment[:TARGET_SAMPLES]

    mel = librosa.feature.melspectrogram(y=segment, sr=TARGET_SR, n_mels=64, fmax=2500)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return log_mel.astype(np.float32)

def process_audio(file_path):
    wav_path = convert_to_wav(file_path)
    y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    y_clean = bandpass_filter(y)

    # Full spectrogram for visual display
    full_mel = librosa.feature.melspectrogram(y=y_clean, sr=TARGET_SR, n_mels=64, fmax=2500)
    full_log_mel = librosa.power_to_db(full_mel, ref=np.max)

    # Slide 3-second windows (1.5s step) across the entire recording
    step = int(TARGET_SAMPLES * 0.5)
    windows = []
    if len(y_clean) <= TARGET_SAMPLES:
        windows.append(make_window_spectrogram(y_clean)[..., np.newaxis])
    else:
        for start in range(0, len(y_clean) - TARGET_SAMPLES + 1, step):
            seg = y_clean[start : start + TARGET_SAMPLES]
            windows.append(make_window_spectrogram(seg)[..., np.newaxis])

    return y_clean, full_log_mel, np.array(windows, dtype=np.float32)