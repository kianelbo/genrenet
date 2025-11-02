import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_acoustic_features(song, offset=0):
    features = []
    y, sr = librosa.load(song, offset=offset, duration=10, mono=True)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma_stft))
    features.append(np.var(chroma_stft))

    rmse = librosa.feature.rms(y=y)
    features.append(np.mean(rmse))
    features.append(np.var(rmse))

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spec_cent))
    features.append(np.var(spec_cent))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spec_bw))
    features.append(np.var(spec_bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.var(zcr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for m in mfcc:
        features.append(np.mean(m))
        features.append(np.var(m))

    return features

temp_file = "save/spec.png"


def create_spectrogram(song, offset=0):
    y, sr = librosa.load(song, offset=offset, duration=10, mono=True)

    s = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    s_db = librosa.amplitude_to_db(s, ref=np.max)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.margins(0, 0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

    librosa.display.specshow(s_db, sr=sr, cmap="gray_r")

    plt.savefig(temp_file, bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open(temp_file).convert("L")
    img = img.resize((128, 128))
    return np.array(img, dtype=np.float32) / 255.0
