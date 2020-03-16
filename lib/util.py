import re

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np

def map_score(perf):
    """ associate a performance midi with a kern score based on filename conventions """
    regex = re.compile('(\d\d\d)_bwv(\d\d\d)(f|p)')
    info = regex.search(perf)
    num, bwv, part = info.group(1,2,3)
    bwv = int(bwv)
    book = 1 + int(bwv > 869)
    score = 'wtc{}{}{:02d}'.format(book,part,bwv - 845 - (book-1)*24)

    return score

def plot_events(ax, events, stride=512, num_windows=2000):
    timings = np.cumsum(events[:,-1])
    x = np.zeros([num_windows,128])
    for i in range(num_windows):
        time = (stride*i)/44100.
        k = np.argmin(time>=timings)
        x[i] = events[k,:128]

    ax.imshow(x.T[::-1][30:90], interpolation='none', cmap='Greys', aspect=num_windows/250)

def colorplot(x, y, figsize):
    cmap = colors.ListedColormap(['white','red','orange','black'])
    bounds = [0,1,2,3,4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=figsize)
    plt.imshow(x.T, interpolation='none', cmap='Greys', aspect=4)
    plt.figure(figsize=figsize)
    plt.imshow(y.T, interpolation='none', cmap='Greys', aspect=4)
    plt.figure(figsize=figsize)
    plt.imshow(x.T*2 + y.T, interpolation='none', cmap=cmap, aspect=4, norm=norm)

def plot_perf_alignment(sig1, sig2, alignment, stride=512, num_windows=None):
    """ plot an alignment of sig2 to sig1 """
    timings1 = np.cumsum(sig1[:,-1])
    timings2 = np.cumsum(sig2[:,-1])
    eps = 1e-12
    
    if num_windows is None: num_windows = int(timings1[-1]*(44100/512.))
    
    x = np.zeros([num_windows,128])
    y = np.zeros([num_windows,128])
    for i in range(num_windows):
        time = (i*stride)/44100.
        k = np.argmin(time>=timings1)
        j = np.argmin(alignment[k]+eps>=timings2)
        
        y[i] = sig2[j,:128]
        x[i] = sig1[k,:128]
        
    colorplot(x[:,30:90][:,::-1], y[:,30:90][:,::-1], (50, 5))

def plot_time_alignment(sig1, sig2, alignment, stride=512, num_windows=None):
    """ plot an alignment of sig2 to time as measured by sig1 """
    timings1 = np.cumsum(sig1[:,-1])
    timings2 = np.cumsum(sig2[:,-1])
    eps = 1e-8
    
    if num_windows is None: num_windows = len(alignment)
    
    x = np.zeros([num_windows,128])
    y = np.zeros([num_windows,128])
    for i in range(num_windows):
        time = (i*stride)/44100.
        k = np.argmin(time>=timings1-eps)
        j = np.argmin(alignment[i]>=timings2-eps)

        y[i] = sig2[j,:128]
        x[i] = sig1[k,:128]
        
    colorplot(x[:,30:90][:,::-1], y[:,30:90][:,::-1], (50, 5))
