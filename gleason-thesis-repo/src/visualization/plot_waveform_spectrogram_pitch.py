import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = "path/to/caregiver_segment.wav"
sr = 16000
snippet_duration = 2.0

y, sr = librosa.load(audio_path, sr=sr, duration=snippet_duration)

D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
S_db = librosa.amplitude_to_db(D, ref=np.max)

f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7")
)
times = librosa.times_like(f0)

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

librosa.display.waveshow(y, sr=sr, ax=ax[0], alpha=0.7)
ax[0].set(title="Waveform (first 2s)", ylabel="Amplitude")

img = librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="log", ax=ax[1])
ax[1].set(title="Log-frequency Spectrogram", ylabel="Freq (Hz)")
fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

ax[2].plot(times, f0, label="Estimated f0", color="r")
ax[2].set(title="Pitch Contour (pYIN)", ylabel="Freq (Hz)", xlabel="Time (s)")
ax[2].legend()

plt.tight_layout()
plt.show()
