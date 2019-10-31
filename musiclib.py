import numpy as np
from scipy import stats
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import sys,time,os
from IPython.display import display, Math, Latex
from mido import MidiFile
import pygame

sentinel = object()
def fft_desc(channel, window_size, stride=sentinel):
    if stride is sentinel:
        stride = window_size/2

    windows = (len(channel) - window_size)/stride
    spectra = np.zeros([windows,window_size])
    for i in xrange(0,windows):
        ts = i*stride
        spectra[i,:] = np.abs(fft(channel[ts:ts+window_size]))
    return spectra

def plotfft(spectra, fs=44100, overlap=2, seconds=5,max_freq=1024,log=True):
    window_size = len(spectra[0])
    wps = overlap*(fs/float(window_size)) # windows per second
    iwps = int(round(wps))
    frames = min(len(spectra[1]),seconds*iwps)

    # most of the interesting visuals are in the low freqs
    view = int(window_size*(max_freq/float(fs)))     

    fig = plt.figure(1,figsize=(7,4))
    if log:
        plt.imshow(np.log(10000 + spectra[0:seconds*iwps,0:view].transpose()),origin='lower',aspect='auto',cmap='Blues')
    else:
        plt.imshow(spectra[0:seconds*iwps,0:view].transpose(),origin='lower',aspect='auto',cmap='Blues')

    return fig

def pick_peaks(feature,window=200,lmbda=1,delta=.1,max_check=10):
    onset_seq = []
    for frame in xrange(0,len(feature)):
        
        # easy rejection
        if feature[frame] < 0:
            continue
    
        history = feature[max(0,frame-window):frame] # I think this is more motivated than centered window;
                                                     # even if we are doing this offline
        if len(history) == 0: history = [0]
        history = np.pad(history,window,mode='median')
    
        # thresholding
        if feature[frame] < delta + lmbda*np.median(history):
            continue
    
        onset = True # looking good so far...
    
        # local max check
        for w in range(max_check):
            if feature[frame] < feature[max(0,frame-w)] or feature[frame] < feature[min(len(feature)-1,frame+w)]:
                onset = False
    
        if onset:
            onset_seq.append(frame)
            
    return onset_seq

def normalize(feature):
    new = feature / np.amax(feature)
    return new - np.mean(new)#

def fonset(feature,sensitivity=1,lockout=10,lookback=200):
    onset_seq = []
    lockcount = lockout
    for frame in xrange(0,len(feature)):
        onset = False # assume no onset
        history = feature[max(0,frame-200):frame] # moving average reference frame
        if len(history) == 0: history = [0]
        history = np.pad(history,200,mode='mean')
        mu = np.mean(history)                    # should we apply decay weight to older history??
        sigma = np.std(history)
    
        if feature[frame] > mu + sensitivity*sigma:
            onset = True
    
        if (onset and lockcount < 0):
            onset_seq.append(frame)
            lockcount = lockout # don't let onsets pile up
        
        lockcount -= 1
        
    return np.array(onset_seq)

# cost is a function of previous alignment and tempo
# see align.ipynb
def cost(midi0,midi1,onset0,onset1,tempo=0,debug=False):
    if tempo == 0:
        return 0 # if no tempo, anything goes
    
    midistep = midi1 - midi0
    elapsed = onset1 - onset0
    
    #cost = 0
    #tolerance = 1.5 
    #if elapsed > tolerance*midistep*tempo or elapsed < (1./tolerance)*midistep*tempo:
    #    cost = 10
    #return cost

    #tolerance = .2
    #if abs(elapsed - midistep*tempo) > tolerance:
    #    cost = 10
    #return cost

    return 5*abs(elapsed - midistep*tempo)**2

# trace back the optimal path starting from alignment (midi,onset)
# midi - the midi event to start the trace from
# onset - the onset to start the trace from
# if depth is set, do a partial trace until we've aligned depth notes
# if include_skips is set, the returned path will declare false +/-
# returns depth alignment pairs (midi,onset) ordered reverse chronologically
def traceback_onsets(ground,observed,P,T,midi,onset,depth=0,include_skips=False):
    j = midi
    k = onset
    A = []
    while depth == 0 or len(A) < depth: #j >= midi - d1 or k >= onset - d2:
        if j >= 0 and k >= 0 and P[j,k] == 1: # match
            A.append([ground[j],observed[k],T[j,k]])
            j -= 1
            k -= 1
        elif j >= 0 and P[j,k] == 2: # false negative
            if include_skips:
                A.append([ground[j],'-'])
            j -= 1
        elif k >= 0: # false positive
            if include_skips:
                A.append(['-',observed[k]])
            k -= 1
        else: # got back to the beginning
            break

    return A

cost_fp = 1
cost_fn = 1

def align_onsets(ground,observed, debug=False, status_updates=False):
    L = np.zeros([len(ground),len(observed)]) # loss matrix
    P = np.zeros(L.shape)                     # path we took
    T = np.zeros(L.shape)                     # implied tempo
    start = 0

    if status_updates:
        start = time.clock()
        print('[----------]', end=" ")
        sys.stdout.flush()

    for j in range(0,len(ground)):
        if status_updates and 0 == j % (len(ground)/10):
            status = '+'*int(round(10*j / float(len(ground))))
            print('\r[' + status.ljust(10,'-') + ']', end=" ")
            sys.stdout.flush()
            
        for k in range(0,len(observed)):
            L_match = 0
            L_fn = cost_fn
            L_fp = cost_fp
            
            if j == 0:
                # best thing to do is just to match
                L_match = k*cost_fp # had to skip k onsets
                L_fp = (k+1)*cost_fp # skipping another onset
                L_fn = cost_fn + k*cost_fp # skipping this midi event
            
            elif k == 0:
                # best thing to do is just to match
                L_match = j*cost_fn # had to skip j midi events
                L_fn = (j+1)*cost_fn # skipping another midi event
                L_fp = cost_fp + j*cost_fn # skipping this onset
                
            else:
                # need two prior alignments to lock in a tempo,
                if j > 1 and k > 1:
                    A = traceback(ground,observed,P,T,j-1,k-1,depth=2)
                    
                    T[j,k] = (A[1][1] - A[0][1])/(A[1][0] - A[0][0])

                    if j > 2 and k > 2:
                        # mix in old tempos with exp decay
                        if A[0][2] == 0: print(A[0])
                        T[j,k] = (T[j,k] + A[0][2]) / 2. # decay faster
                        #T[j,k] = .33*T[j,k] + .67*A[0][2] # decay slower

                    L_match = cost(A[0][0],ground[j],A[0][1],observed[k],tempo=T[j,k],debug=False)
                else:
                    #L_match = cost(ground[j-1],ground[j],observed[k-1],observed[k],debug=False)
                    L_match = 0
                L_match += L[j-1,k-1]
                L_fn = cost_fn + L[j-1,k]
                L_fp = cost_fp + L[j,k-1]
                
            # 1 = match (up and left)
            # 2 = false negative (up)
            # 3 = false positive (left)
            P[j,k] = 1 + np.argmin([L_match,L_fn,L_fp])
            L[j,k] = min(L_match,L_fn,L_fp)
            
    if debug:
        display(Math('\\textbf{Score:}'))
        print(L.round(2))
        display(Math('\\textbf{Path:}'))
        print(P)
        display(Math('\\textbf{Tempo:}'))
        print(T.round(2))
        
    A = traceback(ground,observed,P,T,len(ground)-1,len(observed)-1,include_skips=True)
        
    if status_updates:
        print('\r[++++++++++]')
        print('Elapsed Time: ' + str(time.clock() - start) + 's')
        sys.stdout.flush()

    return list(reversed(A))

def mark_onsets(pcm,fs,onsets,debug=False,stereo=False):
    # create an onset marker
    freq = 440. # concert A in hertz
    mark = 16000*np.sin(freq*2.*np.pi*np.arange(0,1024)/fs)
    mark = mark*np.hanning(len(mark))
    if debug:
        plt.plot(mark)

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

def load_midi(filename):
    midi = MidiFile(filename)

    #print midi.tracks

    midi_onsets = []
    midi_notes = []
    time = 0
    k = 0
    for message in midi:
        time += message.time

        # velocity == 0 equivalent to note_off, see here:
        # http://www.kvraudio.com/forum/viewtopic.php?p=4167096
        if message.type == 'note_on' and message.velocity != 0:
            # some midis seem to have timing info on channel 0
            # but not intended to be played? (e.g. ravel)
            #if message.channel==0:
            #    continue
            if midi_notes != [] and time == midi_onsets[-1]:
                midi_notes[-1].append(message.note)
            else:
                midi_onsets.append(time)
                midi_notes.append([message.note])
    assert time == midi.length

    return np.array(midi_onsets),midi_notes

def playback(clip,num_loops=0,fs=44100,bitdepth=16,stereo=False):
    # the sound module requires data in C layout
    clip = clip.copy(order='C')

    channels = 1
    if stereo:
        channels = 2
    
    max_sample = 2**(bitdepth - 1) - 1

    pygame.mixer.pre_init(frequency=fs,size=-bitdepth,channels=channels)
    pygame.init()
    sound = pygame.sndarray.make_sound(clip)
    snd = sound.play(loops = num_loops)
    pygame.time.delay(int((len(clip)/float(fs))*1000*(1+num_loops)) + 100) # give a few ms to end gracefully
    pygame.quit()

# returns a local noop feature centered at index in data
def feature_noop(data,index,patch_size,normalize=False):
    patch = data[index-patch_size/2:index+patch_size/2]
    assert(len(patch) == patch_size)

    lfeat = patch[:,0]
    rfeat = patch[:,1]

    return (lfeat,rfeat)

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

def wide_feature(data,index,patch_size,stride,width,step_size,normalize=False):
    lfeat,rfeat = feature(data,index,patch_size,normalize)

    wfeat = [lfeat,rfeat]
    for step in range(1,width):
        lfeat,rfeat = feature(data,index - step*step_size*stride,patch_size,normalize)
        wfeat.append(lfeat)
        wfeat.append(rfeat)
        lfeat,rfeat = feature(data,index + step*step_size*stride,patch_size,normalize)
        wfeat.append(lfeat)
        wfeat.append(rfeat)

    flat_feat = np.concatenate(wfeat)

    return flat_feat

def batch_process(process, root, limit = 10000, filetype='.mid'):
    start = time.time()
    processed = 0
    for dirpath,dirnames,filenames in os.walk(root):
        for filename in sorted([f for f in filenames if f.endswith(filetype)]):
            process(os.path.join(dirpath, filename))
            processed += 1
            if processed >= limit: return
    end = time.time()
    print('Finished processing in ' + str(end - start) + ' seconds.')

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
