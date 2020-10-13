import glob
import os
import pickle

import torch

from python_speech_features import logfbank, fbank, sigproc
import numpy as np
import librosa
from tqdm import tqdm


class FeatureExtractor(object):
    def __init__(self, rate):
        self.rate = rate

    def get_features(self, features_to_use, X):
        X_features = None
        accepted_features_to_use = ("logfbank", 'mfcc', 'fbank', 'melspectrogram', 'spectrogram', 'pase')
        if features_to_use not in accepted_features_to_use:
            raise NotImplementedError("{} not in {}!".format(features_to_use, accepted_features_to_use))
        if features_to_use in ('logfbank'):
            X_features = self.get_logfbank(X)
        if features_to_use in ('mfcc'):
            X_features = self.get_mfcc(X,26)
        if features_to_use in ('fbank'):
            X_features = self.get_fbank(X)
        if features_to_use in ('melspectrogram'):
            X_features = self.get_melspectrogram(X)
        if features_to_use in ('spectrogram'):
            X_features = self.get_spectrogram(X)
        if features_to_use in ('pase'):
            X_features = self.get_Pase(X)
        return X_features

    def get_logfbank(self, X):
        def _get_logfbank(x):
            out = logfbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000,
                           nfilt=40)
            return out

        X_features = np.apply_along_axis(_get_logfbank, 1, X)
        return X_features

    def get_mfcc(self, X, n_mfcc=13):
        def _get_mfcc(x):
            mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
            # delta = librosa.feature.delta(mfcc_data)
            # delta_delta = librosa.feature.delta(mfcc_data, order=2)
            # mfcc_data = np.expand_dims(mfcc_data, 0)
            # delta = np.expand_dims(delta, 0)
            # delta_delta = np.expand_dims(delta_delta, 0)
            # out = np.concatenate((mfcc_data, delta, delta_delta), 0)
            return mfcc_data

        X_features = np.apply_along_axis(_get_mfcc, 1, X)
        return X_features

    def get_fbank(self, X):
        def _get_fbank(x):
            out, _ = fbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024)
            return out

        X_features = np.apply_along_axis(_get_fbank, 1, X)
        return X_features

    def get_melspectrogram(self, X):
        def _get_melspectrogram(x):
            mel = librosa.feature.melspectrogram(y=x, sr=self.rate, n_fft=800, hop_length=400)[np.newaxis, :]
            out=np.log10(mel).squeeze()
            return out

        X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
        return X_features

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features

    # def get_Pase(self, X):
    #     pase = wf_builder('PASE/cfg/PASE.cfg')
    #     pase.eval()
    #     pase.load_pretrained('PASE/PASE.ckpt', load_last=True, verbose=True)
    #     # tq=tqdm(total=X.shape[0])
    #     def _get_spectrogram(x):
    #         x = torch.from_numpy(x)
    #         x = x.unsqueeze(0).unsqueeze(0)
    #         y = pase(x).detach().numpy()
    #         # tq.update(1)
    #         return y
    #
    #     X_features = np.apply_along_axis(_get_spectrogram, 1, X)
    #     return X_features
    def get_Pase(self,X):
        return X

