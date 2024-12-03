import librosa
import numpy as np
from scipy import signal

def preprocess_audio(file_path, sr=22050, normalize=True, remove_noise=True, 
                    trim_silence=True, apply_compression=True, improve_snr=True):
    """
    Preprocess audio file with enhanced SNR improvement.
    """
    audio, sr = librosa.load(file_path, sr=sr)
    
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=30)
    
    if improve_snr:
        audio = improve_signal_to_noise(audio, sr)
    
    if remove_noise:
        audio = spectral_gate(audio)
    
    if normalize:
        audio = librosa.util.normalize(audio)
    
    if apply_compression:
        audio = apply_dynamic_compression(audio)
    
    return audio, sr

def improve_signal_to_noise(audio, sr, frame_length=2048, hop_length=512):
    """
    Improve SNR using spectral subtraction and Wiener filtering.
    """
    # Spectral subtraction
    D = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    mag = np.abs(D)
    phase = np.angle(D)
    
    # Estimate noise spectrum from relatively silent frames
    noise_spec = estimate_noise_spectrum(mag)
    
    # Apply spectral subtraction
    mag_subtracted = np.maximum(mag - noise_spec[:, np.newaxis], 0)
    
    # Wiener filtering
    H = mag_subtracted**2 / (mag_subtracted**2 + noise_spec[:, np.newaxis]**2 + 1e-6)
    mag_filtered = mag * H
    
    # Reconstruct signal
    D_enhanced = mag_filtered * np.exp(1j * phase)
    audio_enhanced = librosa.istft(D_enhanced, hop_length=hop_length)
    
    # Apply median filtering to reduce musical noise
    audio_enhanced = signal.medfilt(audio_enhanced, kernel_size=3)
    
    return audio_enhanced

def estimate_noise_spectrum(mag_spec, p=0.15):
    """
    Estimate noise spectrum from magnitude spectrogram.
    """
    # Sort magnitudes and take the lowest p% as noise estimate
    sorted_mag = np.sort(mag_spec, axis=1)
    n_frames = sorted_mag.shape[1]
    n_noise_frames = int(n_frames * p)
    noise_estimate = np.mean(sorted_mag[:, :n_noise_frames], axis=1)
    return noise_estimate

def spectral_gate(audio, thresh_n=2.0):
    """
    Apply spectral gating for noise reduction.
    """
    D = librosa.stft(audio)
    mag = np.abs(D)
    phase = np.angle(D)
    
    noise_floor = np.mean(np.min(mag, axis=1))
    thresh = thresh_n * noise_floor
    
    mask = mag > thresh
    mag = mag * mask
    
    D = mag * np.exp(1j * phase)
    audio = librosa.istft(D)
    
    return audio

def apply_dynamic_compression(audio, threshold=-20, ratio=4, attack_ms=5, release_ms=50):
    """
    Apply dynamic range compression.
    """
    attack_time = attack_ms / 1000
    release_time = release_ms / 1000
    
    env = np.zeros_like(audio)
    alpha_a = np.exp(-1 / (sr * attack_time))
    alpha_r = np.exp(-1 / (sr * release_time))
    
    for i in range(1, len(audio)):
        env[i] = max(abs(audio[i]), 
                    alpha_a * env[i-1] if abs(audio[i]) > env[i-1] 
                    else alpha_r * env[i-1])
    
    env_db = 20 * np.log10(env + 1e-8)
    gain_db = np.minimum(0, (threshold - env_db) * (1 - 1/ratio))
    gain_linear = np.power(10, gain_db/20)
    
    return audio * gain_linear

if __name__ == "__main__":
    file_path = "input.wav"
    processed_audio, sr = preprocess_audio(file_path, improve_snr=True)
    librosa.output.write_wav("output.wav", processed_audio, sr)
