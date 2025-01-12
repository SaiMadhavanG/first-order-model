import torchaudio
import numpy as np
from scipy import interpolate
import torch.nn as nn
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

class VectorToImage(nn.Module):
    def __init__(self, input_dim=1024, seq_len=5, target_size=64):
        super(VectorToImage, self).__init__()
        self.seq_len = seq_len
        self.target_size = target_size

        # Fully connected layer to reduce the feature dimension
        self.fc = nn.Sequential(
            nn.Linear(input_dim, target_size * target_size),  # Project 1024 to 64x64
            nn.ReLU()
        )

        # Final convolution to aggregate sequence dimension into 1 channel
        self.conv = nn.Conv2d(seq_len, 1, kernel_size=1)  # Aggregate 5 -> 1 channel

    def forward(self, x):
        """
        Input:
            x: (b, 5, 1024)
        Output:
            x: (b, 1, 64, 64)
        """
        b, seq_len, feature_dim = x.shape

        # Apply fully connected layer to each sequence element
        x = x.view(b * seq_len, feature_dim)  # Flatten sequence for processing: (b*5, 1024)
        x = self.fc(x)  # Project to (b*5, 64*64)
        x = x.view(b, seq_len, self.target_size, self.target_size)  # Reshape to (b, 5, 64, 64)

        # Aggregate sequence dimension into 1 channel
        x = self.conv(x)  # Output shape: (b, 1, 64, 64)

        return x

class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(self, device, sample_rate):
        super(Wav2Vec2FeatureExtractor, self).__init__()
        self.device = device
        self.bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        self.model = self.bundle.get_model().to(self.device)
        self.num_frames = 5
        self.sample_rate = sample_rate
        self.projector = VectorToImage(input_dim=1024, seq_len=self.num_frames, target_size=64).to(self.device)

    def forward(self, waveform):
        waveform = waveform.to(self.device)
        
        if self.sample_rate != self.bundle.sample_rate:
            resampler = torchaudio.transforms.Resample(self.sample_rate, self.bundle.sample_rate).to(self.device)
            waveform = resampler(waveform)
            
        
        features, _ = self.model.extract_features(waveform)
        features = features[-1].cpu().numpy()
        num_frames_audio = features.shape[1]
        x_old = np.linspace(0, num_frames_audio - 1, num_frames_audio)
        x_new = np.linspace(0, num_frames_audio - 1, self.num_frames)
        embeddings_resampled = np.zeros((1, self.num_frames, features.shape[2]))
        for i in range(features.shape[2]):
            f = interpolate.interp1d(x_old, features[0, :, i], kind='cubic')
            embeddings_resampled[0, :, i] = f(x_new)

        embeddings_resampled = torch.tensor(embeddings_resampled).float().to(self.device)
        output = self.projector(embeddings_resampled)
        
        return output

class AudioToMelVector(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=256, hop_length=210, n_mels=80, target_shape=(256, 64, 64)):
        """
        Module to process an audio waveform into a mel spectrogram and then transform it into a vector of target shape.

        Args:
            sample_rate (int): Sample rate of the input audio waveform.
            n_fft (int): Number of FFT bins for the STFT.
            hop_length (int): Hop length for STFT (controls time resolution).
            n_mels (int): Number of mel frequency bins.
            target_shape (tuple): Desired output shape, default is (256, 64, 64).
        """
        super(AudioToMelVector, self).__init__()
        self.target_shape = target_shape
        self.sample_rate = sample_rate

        # MelSpectrogram transformation
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=80)

        # Transformation layers to reshape to target_shape
        self.projection = nn.Sequential(
            nn.Conv2d(1, target_shape[0], kernel_size=3, stride=1, padding=1),  # Expand to 256 channels
            nn.ReLU(),
            nn.Conv2d(target_shape[0], target_shape[0], kernel_size=3, stride=1, padding=1),  # Stabilize
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((target_shape[1], target_shape[2]))  # Resize to (64, 64)

    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Audio waveform of shape (batch_size, num_channels, num_samples).
                                     Expecting mono audio (num_channels=1).

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, 256, 64, 64).
        """
        # Ensure the input is mono by averaging channels if needed
        if waveform.ndim == 3 and waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)  # Convert to mono

        # Step 1: Compute the mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)  # Shape: (batch_size, n_mels, time_frames)
        mel_spec_db = self.amplitude_to_db(mel_spec)  # Convert to decibels

        # Add a channel dimension for Conv2d (batch_size, 1, n_mels, time_frames)
        mel_spec_db = mel_spec_db.unsqueeze(1)

        # Step 2: Project to target shape
        projected = self.projection(mel_spec_db)  # Expand to 256 channels
        output = self.pool(projected)  # Resize to (batch_size, 256, 64, 64)

        return output