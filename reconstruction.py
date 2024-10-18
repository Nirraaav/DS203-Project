import librosa
import numpy as np
import soundfile as sf
import time

def mfcc_to_audio(mfcc, sr=44100, hop_length=512, n_mels=128, n_fft=2048, iterations=100):
    mfcc = np.asarray(mfcc)
    mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(mfcc, n_mels=n_mels)    
    audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr, hop_length=hop_length, n_fft=n_fft, n_iter=iterations)
    return audio

for i in range(1, 116):
    curr_time = time.time()
    mfcc = np.loadtxt('data/' + ('0' if i < 10 else '') + str(i) + '-MFCC.csv', delimiter=',')
    reconstructed_audio = mfcc_to_audio(mfcc)
    sf.write('reconstructed-data/' + str(i) + '.wav', reconstructed_audio, 44100)

    print('Reconstructed audio saved as reconstructed-data-' + str(i) + '.wav at time ' + str(time.time() - curr_time) + ' seconds')

