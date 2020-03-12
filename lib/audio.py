import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

def mark_onsets(pcm,fs,onsets,stereo=False):
    # create an onset marker
    freq = 440. # concert A in hertz
    mark = 16000*np.sin(freq*2.*np.pi*np.arange(0,1024)/fs)
    mark = mark*np.hanning(len(mark))

    out = np.zeros(pcm.shape)
    for onset in onsets:
        if onset + len(mark) < len(out):
            try:
                if stereo:
                    out[onset:onset+len(mark)] = np.vstack([mark,mark]).T
                else:
                    out[onset:onset+len(mark)] = mark
            except ValueError:
                print(onset)
                print(onset + len(mark))
    out = .2*pcm + .8*out

    return out.astype(np.int16)

# returns a local spectrogram feature centered at index in data
def feature_spectrogram(data,index,patch_size,normalize=False):
    patch = data[index-patch_size/2:index+patch_size/2]
    assert(len(patch) == patch_size)

    lfeat = np.abs(fft(patch[:,0]))[0:patch_size/2]
    rfeat = np.abs(fft(patch[:,1]))[0:patch_size/2]
 
    return (lfeat,rfeat)

# returns a local log-spectrogram feature centered at index in data
def feature_logspectrogram(data,index,patch_size,normalize=False):
    patch = data[index-patch_size/2:index+patch_size/2]
    assert(len(patch) == patch_size)

    lfeat = np.abs(fft(patch[:,0]))[0:patch_size/2]
    rfeat = np.abs(fft(patch[:,1]))[0:patch_size/2]
 
    lfeat = np.log(lfeat + .0001)
    rfeat = np.log(rfeat + .0001)

    return (lfeat,rfeat)

# returns a local ReLU log-spectrogram feature centered at index in data
def feature(data,index,patch_size,normalize=False):
    patch = data[int(index-patch_size//2):int(index+patch_size//2)]
    assert(len(patch) == patch_size)
    
    lfeat = np.abs(fft(patch[:,0]))[0:patch_size//2]
    rfeat = np.abs(fft(patch[:,1]))[0:patch_size//2]
    
    lfeat = np.log(lfeat + .0001)
    rfeat = np.log(rfeat + .0001)
        
    lfeat = lfeat - np.mean(lfeat)
    rfeat = rfeat - np.mean(rfeat)
                
    lfeat = lfeat.clip(min=0)
    rfeat = rfeat.clip(min=0)
    
    if normalize:
        lfeat = lfeat / (100. + np.linalg.norm(lfeat))
        rfeat = rfeat / (100. + np.linalg.norm(rfeat))

    return (lfeat,rfeat)

def featurize(data,fs,f,patch_size,stride=512,normalize=False):
    num_windows = len(data)//stride - patch_size//stride + 1
    num_features = len(f(data,stride+patch_size//2,patch_size,normalize)[0])

    rep = np.zeros((num_windows,num_features))
    # skip the last window (we will overflow if stride doesn't divide window size)
    for window in range(num_windows-1):
        L,R = f(data,stride*window+patch_size//2,patch_size,normalize)
        rep[window,:] = (L+R)/2.

    return rep.T

def mark_notes(y,onsets,notes,mix_size=4096,fs=44100):
    out = np.zeros(y.shape)
    for onset,note_block in zip(onsets,notes):
        for note in note_block:
            freq = 440.*2**((note - 69.)/12.)
            mark = 32000*np.sin(freq*2.*np.pi*np.arange(0,mix_size)/fs)/len(note_block)
            #mark = mark*np.hanning(len(mark))
            sample = int(round(onset))
            if sample + len(mark) < len(out):
                out[sample:sample+mix_size] += mark
    return out

def mark_notes_with_offsets(y, notes_onsets_offsets, mix_size=4096, fs=44100):
    out = np.zeros(y.shape)
    for note, onset, offset in notes_onsets_offsets:
        freq = 440.*2**((note - 69.)/12.)
        mark = 32000*np.sin(freq*2.*np.pi*np.arange(0,mix_size)/fs)
        sample_onset = int(round(onset))
        sample_offset = int(round(offset))
        if sample_onset + len(mark) < len(out):
            out[sample_onset:sample_onset+mix_size] += mark
        if sample_offset + len(mark) < len(out):
            out[sample_offset:sample_offset+mix_size] += mark
    return out

def create_filters(d):
    x = np.linspace(0, 2*np.pi, d, endpoint=False)
    mask = 0.5-0.5*np.cos(x)
    wsin = np.empty((128,d), dtype=np.float32)
    wcos = np.empty((128,d), dtype=np.float32)
    for i in range(128):
        wsin[i,:] = mask*np.sin((d/44100.)*440.*2**((i-69)/12.)*x)
        wcos[i,:] = mask*np.cos((d/44100.)*440.*2**((i-69)/12.)*x)

    return torch.Tensor(wsin),torch.Tensor(wcos)
