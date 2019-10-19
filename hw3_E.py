import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import read as wavread
from ass3solution import *


def make_sin(fs, freq, seconds):
    time_in_sec = np.arange(fs*seconds)/fs
    radians = time_in_sec * freq * 2 * np.pi
    sin = np.sin(radians)
    return sin


def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)

    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2**(nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    return (samplerate, audio)



def convert_freq2midi(fInHz, fA4InHz=440):
    def convert_freq2midi_scalar(f, fA4InHz):

        if f <= 0:
            return 0
        else:
            return (69 + 12 * np.log2(f / fA4InHz))
        fInHz = np.asarray(fInHz)
        if fInHz.ndim == 0:
            return convert_freq2midi_scalar(fInHz, fA4InHz)

    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
        return convert_freq2midi_scalar(fInHz, fA4InHz)

    midi = np.zeros(fInHz.shape)
    for k, f in enumerate(fInHz):
        midi[k] = convert_freq2midi_scalar(f, fA4InHz)

    return (midi)


def eval_pitchtrack(estimateInHz, groundtruthInHz):
    if np.abs(groundtruthInHz).sum() <= 0:
        return 0

    # truncate longer vector
    if groundtruthInHz.size > estimateInHz.size:
        estimateInHz = estimateInHz[np.arange(0, groundtruthInHz.size)]
    elif estimateInHz.size > groundtruthInHz.size:
        groundtruthInHz = groundtruthInHz[np.arange(0, estimateInHz.size)]

    diffInCent = 100 * (convert_freq2midi(estimateInHz) - convert_freq2midi(groundtruthInHz))

    rms = np.sqrt(np.mean(diffInCent[groundtruthInHz != 0] ** 2))
    return (rms)


def run_evaluation1(complete_path_to_data_folder):


    # init
    rmsAvg_1 = 0
    rmsAvg_2 = 0
    iNumOfFiles = 0

    # for loop over files
    for file in os.listdir(complete_path_to_data_folder):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = ToolReadAudio(complete_path_to_data_folder + file)

            # read ground truth (assume the file is there!)
            refdata = np.loadtxt(complete_path_to_data_folder + os.path.splitext(file)[0] + '.f0.Corrected.txt')
        else:
            continue

        # extract pitch
        [f0, t] = track_pitch_fftmax(afAudioData, 1024, 512, fs)
        [f0_2, t] = track_pitch_hps(afAudioData, 1024, 512, fs)
        [f0_3, t] = track_pitch(afAudioData, 1024, 512, fs)
        # compute rms and accumulate
        rmsAvg_1 += eval_pitchtrack_v2(f0, refdata[:, 2])
        rmsAvg_2 += eval_pitchtrack_v2(f0_2, refdata[:, 2])
        rmsAvg = np.concatenate([rmsAvg_1, rmsAvg_2])

    if iNumOfFiles == 0:
        return -1

    return rmsAvg / iNumOfFiles



def executeassign3():

    # Question E.1

    fs = 44100
    sin441 = make_sin(fs, 441, 1)
    sin882 = make_sin(fs, 882, 1)
    blockSize = 1024
    hopSize = 512

    sin = np.concatenate([sin441, sin882])
    xb, t = block_audio(sin, blockSize, hopSize, fs)

    f0, timeInSec = track_pitch_fftmax(xb, blockSize, hopSize, fs)
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(f0, timeInSec)
    plt.xlabel('Time (s)')
    plt.ylabel('F0 (Hz)')

    error441 = np.abs(f0[:len(f0)//2]-441*np.ones(len(f0//2)))
    error882 = np.abs(f0[len(f0)//2:]-882*np.ones(len(f0//2)))
    error = np.concatenate([error441, error882])

    plt.subplot(1, 2, 2)
    plt.plot(error, timeInSec)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (Hz)')

    track_pitch_hps(xb, blockSize, hopSize, fs)
    plt.figure(2)
    plt.plot(track_pitch_hps[0], track_pitch_hps[1])

    # Question E.2

    blockSize_2 = 2048
    hopSize_2 = 512
    xb_2, t_2 = block_audio(sin, blockSize_2, hopSize_2, fs)
    f0_2, timeInSec_2 = track_pitch_fftmax(xb_2, blockSize_2, hopSize_2, fs)
    plt.subplot(1, 2, 1)
    plt.plot(f0_2, timeInSec_2)
    plt.xlabel('Time (s)')
    plt.ylabel('F0_2 (Hz)')

    error441_2 = np.abs(f0[:len(f0_2) // 2] - 441 * np.ones(len(f0_2 // 2)))
    error882_2 = np.abs(f0[len(f0_2) // 2:] - 882 * np.ones(len(f0_2 // 2)))
    error_2 = np.concatenate([error441_2, error882_2])

    plt.subplot(1, 2, 2)
    plt.plot(error_2, timeInSec_2)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (Hz)')

    # Question E3,4,5,6
    run_evaluation1()
    run_evaluation2()
    return ()

    # Question E5
def track_pitch(x, blockSize, hopSize, fs, method, voicingThres):
    if method == 'acf':
        f0, timeInSec = track_pitch_acf(x,blockSize, hopSize, fs)
    elif method == 'hps':
        f0, timeInSec = track_pitch_hps(x,blockSize, hopSize, fs)
    elif method == 'max':
        f0, timeInSec = track_pitch_fftmax(x,blockSize, hopSize, fs)

    rmsDb = extract_rms(x)
    mask = create_voicing_mask(rmsDb, voicingThres)
    f0Adj = apply_voicing_mask(f0, mask)

    return f0Adj, timeInSec

    # Question E6

def run_evaluation2(complete_path_to_data_folder):

    # init
    rmsAvg_1 = 0
    rmsAvg_2 = 0
    rmsAvg_3 = 0
    rmsAvg_4 = 0
    rmsAvg_5 = 0
    rmsAvg_6 = 0
    iNumOfFiles = 0

    # for loop over files
    for file in os.listdir(complete_path_to_data_folder):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = ToolReadAudio(complete_path_to_data_folder + file)

            # read ground truth (assume the file is there!)
            refdata = np.loadtxt(complete_path_to_data_folder + os.path.splitext(file)[0] + '.f0.Corrected.txt')
        else:
            continue

        # extract pitch
        [f0_1, t] = track_pitch(afAudioData, 1024, 512, fs, 'acf', -40)
        [f0_2, t] = track_pitch(afAudioData, 1024, 512, fs, 'hps', -40)
        [f0_3, t] = track_pitch(afAudioData, 1024, 512, fs, 'max', -40)
        [f0_4, t] = track_pitch(afAudioData, 1024, 512, fs, 'acf', -20)
        [f0_5, t] = track_pitch(afAudioData, 1024, 512, fs, 'hps', -20)
        [f0_6, t] = track_pitch(afAudioData, 1024, 512, fs, 'max', -20)
        # compute rms and accumulate
        rmsAvg_1 += eval_pitchtrack_v2(f0_1, refdata[:, 2])
        rmsAvg_2 += eval_pitchtrack_v2(f0_2, refdata[:, 2])
        rmsAvg_3 += eval_pitchtrack_v2(f0_3, refdata[:, 2])
        rmsAvg_4 += eval_pitchtrack_v2(f0_4, refdata[:, 2])
        rmsAvg_5 += eval_pitchtrack_v2(f0_5, refdata[:, 2])
        rmsAvg_6 += eval_pitchtrack_v2(f0_6, refdata[:, 2])

        rmsAvg = np.concatenate([rmsAvg_1, rmsAvg_2, rmsAvg_3, rmsAvg_4, rmsAvg_5, rmsAvg_6])

    if iNumOfFiles == 0:
        return -1

    return rmsAvg / iNumOfFiles


