import time
import warnings

import librosa
import pitch
import parselmouth

import scipy
import scipy.fftpack as fftpk
import numpy as np
import scipy.io.wavfile as wavfile
import sklearn

from pyrpde import rpde
from parselmouth.praat import call
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class FeatureHelper:
    def extract_mfcc(self):
        y = librosa.load(audioFile, sr=16000)[0]
        S = librosa.feature.melspectrogram(y=y, n_mels=64, n_fft=320, hop_length=160)
        norm_log_S = np.clip((librosa.power_to_db(S, ref=np.max)+100) / 100, 0, 1)

        return norm_log_S

    def extract_dft(self):
        y, sr = librosa.load(audioFile, sr=None)

        x = np.fft.fft(y)
        return x

    def extract_jitter(self):
        j = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

        return j

    def extract_shimmer(self):
        sound = parselmouth.Sound(audioFile)
        shimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        return shimmer

    def extract_HNR(self):
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        return hnr

    def extract_PPT(self):
        y, sr = librosa.load(audioFile)
        pitch = librosa.pitch_tuning(y)
        power = librosa.power(y)
        time = librosa.get_duration(y, sr=sr)

        return pitch, power, time

    def extract_DFA(self):
        window_sizes = [10, 20, 30]
        rate, data = wavfile.read(audioFile)
        cumulative_sum = np.cumsum(data - np.mean(data))

        fluctuations = []
        for window_size in window_sizes:
            num_segments = len(cumulative_sum) // window_size

            segment_means = np.zeros(num_segments)
            for i in range(num_segments):
                segment = cumulative_sum[i * window_size: (i + 1) * window_size]
                segment_means[i] = np.mean(segment)

            segment_diffs = np.abs(np.cumsum(segment_means - cumulative_sum[:num_segments * window_size:window_size]))

            fluctuation = np.sqrt(np.mean(segment_diffs ** 2))
            fluctuations.append(fluctuation)

        return fluctuations

    def extract_RPDE(self):
        rate, data = wavfile.read(audioFile)
        entropy, histogram = rpde(data, tau=30, dim=4, epsilon=0.01)

        return entropy

    def extract_all_features(self, file, id):
        global audioFile, sound, pointProcess, f0min, f0max
        audioFile = file
        # sound = parselmouth.Sound(audioFile)
        # f0min = 75
        # f0max = 500
        # pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

        # start_process = time.time()

        mfccs = self.extract_mfcc()

        return mfccs

        # start = time.time()
        #
        # dfts = self.extract_dft()
        # print("==========Extracted : DFT of %s==========" % id)
        # print('ETA : %.5fs' % (time.time() - start))
        #
        # start = time.time()
        #
        # jitters = self.extract_jitter()
        # print("==========Extracted : Jitter of %s==========" % id)
        # print('ETA : %.5fs' % (time.time() - start))
        #
        # start = time.time()
        #
        # shimmers = self.extract_shimmer()
        # print("==========Extracted : Shimmer of %s==========" % id)
        # print('ETA : %.5fs' % (time.time() - start))
        #
        # start = time.time()
        #
        # hnrs = self.extract_HNR()
        # print("==========Extracted : HNR of %s==========" % id)
        # print('ETA : %.5fs' % (time.time() - start))
        #
        # start = time.time()
        #
        # pitch, power, t = self.extract_PPT()
        # print("==========Extracted : PPT of %s==========" % id)
        #
        # print('ETA : %.5fs' % (time.time() - start))
        #
        # start = time.time()
        #
        # rpdes = self.extract_RPDE()
        # print("==========Extracted : RPDE of %s==========" % id)
        # print('ETA : %.5fs' % (time.time() - start))
        #
        # start = time.time()
        # dfas = self.extract_DFA()
        # print("==========Extracted : DFA of %s==========" % id)
        # print('ETA : %.5fs' % (time.time() - start))
        #
        # end = time.time()
        # print('all features extracted successfully! patient ID : %s, ETA : %.5f' % (id, end - start_process))
        #
        # return [mfccs, dfts, jitters, shimmers, hnrs, pitch, power, t, rpdes, dfas]