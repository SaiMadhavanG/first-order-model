import os
import torch
import numpy as np
import cv2
import torchaudio
from scipy import interpolate
from moviepy.editor import AudioFileClip, VideoFileClip
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from tqdm import tqdm

class AudioHandler:
    def __init__(self, device="cpu"):
        self.device = device
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
        self.model = self.bundle.get_model().to(self.device)

    def process(self, audio, num_frames):
        waveform, sample_rate = torchaudio.load(audio)
        waveform = waveform.to(self.device)
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.bundle.sample_rate)(waveform)
        with torch.no_grad():
            features, _ = self.model.extract_features(waveform)
            features = features[-1].cpu().numpy()
            num_frames_audio = features.shape[1]
            x_old = np.linspace(0, num_frames_audio - 1, num_frames_audio)
            x_new = np.linspace(0, num_frames_audio - 1, num_frames)
            embeddings_resampled = np.zeros((1, num_frames, features.shape[2]))
            for i in range(features.shape[2]):
                f = interpolate.interp1d(x_old, features[0, :, i], kind='cubic')
                embeddings_resampled[0, :, i] = f(x_new)
        return embeddings_resampled
    
audio_handler = AudioHandler('cuda')

def process_audio(audio_path, num_frames):
    return audio_handler.process(audio_path, num_frames)

