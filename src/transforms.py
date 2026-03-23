import torchaudio
import torch
import torch.nn as nn
    
class AudioTransform(nn.Module):
    def __init__(self, target_sr, target_samples, n_fft, hop_length, n_mels):
        super().__init__()
        self.target_sr = target_sr
        self.target_samples = target_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def forward(self, signal, sr):
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            signal = resampler(signal)

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        if signal.shape[1] > self.target_samples:
            signal = signal[:, :self.target_samples]
        elif signal.shape[1] < self.target_samples:
            padding = self.target_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, padding))

        return self.mel_spectrogram(signal)
