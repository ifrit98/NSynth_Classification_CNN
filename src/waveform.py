import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram


def plot_wave(instr_family_names, samples_arr, pred, is_correct, sr=16000):
    fig = plt.figure()
    n = ''
    for n, f in zip(instr_family_names, samples_arr):
        f = f.numpy().flatten()
        display.waveplot(f, sr=sr)
    plt.title(instr_family_names[0] + ' Waveplot, Correctly Classified: ' + str(is_correct), x=0.5, y=0.915, fontsize=8)
    if is_correct:
        fig.savefig('./graphs/' + str(n) + '_waveform_c.png')
    else:
        fig.savefig('./graphs/' + str(n) + '_waveform_inc_' + str(pred) + '.png')


def plot_specgram(instr_family_names, samples_arr, pred, is_correct, sr=16000):
    fig = plt.figure()
    n = ''
    for n, f in zip(instr_family_names, samples_arr):
        f = f.numpy().flatten()
        specgram(f, Fs=sr)
    plt.title(instr_family_names[0] + ' Spectrogram, Correctly Classified: ' + str(is_correct), x=0.5, y=0.915, fontsize=8)
    if is_correct:
        fig.savefig('./graphs/' + str(n) + '_specgram_c.png')
    else:
        fig.savefig('./graphs/' + str(n) + '_specgram_inc_ ' + str(pred) + '.png')


def plot_log_power_spec(instr_family_names, samples_arr, pred, is_correct):
    fig = plt.figure()
    n = ''
    for n, f in zip(instr_family_names, samples_arr):
        f = f.numpy().flatten()
        D = librosa.core.amplitude_to_db(np.abs(librosa.stft(f)) ** 2, ref=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
    plt.title(instr_family_names[0] + 'Log power spectrogram Correctly Classified: ' + str(is_correct), x=0.5, y=0.915, fontsize=8)
    if is_correct:
        fig.savefig('./graphs/' + str(n) + '_log_power_c.png')
    else:
        fig.savefig('./graphs/' + str(n) + '_log_power_inc_' + str(pred) + '.png')
