import psycopg2
import numpy as np
from scipy.io import wavfile
from intervaltree import Interval,IntervalTree

# database credentials
database = 'music'
user = 'thickstn'

bach_train_small = {
    2682 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-1pre.wav',
    2683 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-2all.wav',
    2684 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-3cou.wav',
    2685 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-4sar.wav',
    2686 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-5men.wav',
}

bach_test_small = {
    2687 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-6gig.wav',
}

bach_train = {
    2682 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-1pre.wav',
    2683 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-2all.wav',
    2684 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-3cou.wav',
    2685 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-4sar.wav',
    2686 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-5men.wav',
    2687 : '/home/thickstn/dbfiles/Bach/BWV1007/2/cs1-6gig.wav',
    2688 : '/home/thickstn/dbfiles/Bach/BWV1008/1/cs2-1pre.wav',
    2689 : '/home/thickstn/dbfiles/Bach/BWV1008/1/cs2-2all.wav',
    2690 : '/home/thickstn/dbfiles/Bach/BWV1008/1/cs2-3cou.wav',
    2691 : '/home/thickstn/dbfiles/Bach/BWV1008/1/cs2-4sar.wav',
    2692 : '/home/thickstn/dbfiles/Bach/BWV1008/1/cs2-5men.wav',
    2693 : '/home/thickstn/dbfiles/Bach/BWV1008/1/cs2-6gig.wav',
    2700 : '/home/thickstn/dbfiles/Bach/BWV1010/2/cs4-1pre.wav',
    2701 : '/home/thickstn/dbfiles/Bach/BWV1010/2/cs4-2all.wav',
    2702 : '/home/thickstn/dbfiles/Bach/BWV1010/2/cs4-3cou.wav',
    2703 : '/home/thickstn/dbfiles/Bach/BWV1010/2/cs4-4sar.wav',
    2704 : '/home/thickstn/dbfiles/Bach/BWV1010/2/cs4-5bou.wav',
    2705 : '/home/thickstn/dbfiles/Bach/BWV1010/2/cs4-6gig.wav',
    2706 : '/home/thickstn/dbfiles/Bach/BWV1011/2/cs5-1pre.wav',
    2707 : '/home/thickstn/dbfiles/Bach/BWV1011/2/cs5-2all.wav',
    2708 : '/home/thickstn/dbfiles/Bach/BWV1011/2/cs5-3cou.wav',
    2709 : '/home/thickstn/dbfiles/Bach/BWV1011/2/cs5-4sar.wav',
    2710 : '/home/thickstn/dbfiles/Bach/BWV1011/2/cs5-5gav.wav',
    2711 : '/home/thickstn/dbfiles/Bach/BWV1011/2/cs5-6gig.wav',
    2712 : '/home/thickstn/dbfiles/Bach/BWV1012/2/cs6-1pre.wav',
    2713 : '/home/thickstn/dbfiles/Bach/BWV1012/2/cs6-2all.wav',
    2714 : '/home/thickstn/dbfiles/Bach/BWV1012/2/cs6-3cou.wav',
    2715 : '/home/thickstn/dbfiles/Bach/BWV1012/2/cs6-4sar.wav',
    2716 : '/home/thickstn/dbfiles/Bach/BWV1012/2/cs6-5gav.wav',
    2717 : '/home/thickstn/dbfiles/Bach/BWV1012/2/cs6-6gig.wav',
}

bach_test = {
    2694 : '/home/thickstn/dbfiles/Bach/BWV1009/3/cs3-1pre.wav',
    2695 : '/home/thickstn/dbfiles/Bach/BWV1009/3/cs3-2all.wav',
    2696 : '/home/thickstn/dbfiles/Bach/BWV1009/3/cs3-3cou.wav',
    2697 : '/home/thickstn/dbfiles/Bach/BWV1009/3/cs3-4sar.wav',
    2698 : '/home/thickstn/dbfiles/Bach/BWV1009/3/cs3-5bou.wav',
    2699 : '/home/thickstn/dbfiles/Bach/BWV1009/3/cs3-6gig.wav',
}

def featurize(data,fs,f,patch_size,stride=512,normalize=False):
    num_windows = len(data)//stride - patch_size//stride + 1
    num_features = len(f(data,stride+patch_size//2,patch_size,normalize)[0])

    rep = np.zeros((num_windows,num_features))
    # skip the last window (we will overflow if stride doesn't divide window size)
    for window in range(num_windows-1): 
        L,R = f(data,stride*window+patch_size//2,patch_size,normalize)
        rep[window,:] = (L+R)/2.

    return rep.T

def load_data(handles,f,window=2048,stride=512,num_labels=128,mono=False):
    X = []
    Y = []
    conn = psycopg2.connect("dbname={} user={}".format(database,user))
    cur = conn.cursor()
    for id in handles.keys():
        print('+', end=" ")
        fs, record = wavfile.read(handles[id])
        assert fs == 44100
        
        representation = featurize(record/16000.,fs,f,patch_size=window,stride=stride)
        num_features = representation.shape[0]
        representation = np.concatenate((representation,np.zeros((num_features,1*fs/stride))),axis=1) # pad out
        X.append(representation.T)
        
        cur.callproc('get_labels', ['labels',id])
        cur2 = conn.cursor('labels')
        labels = np.zeros(shape=(representation.shape[1],num_labels)) # zero is silence
        for label in cur2:
            start_time = float(label[0])
            end_time = float(label[1])
            note = label[2]

            for frame in range(int(start_time*fs/stride),int(end_time*fs/stride)):
                labels[frame,note] = 1
        Y.append(labels)
                
        cur2.close()
    conn.close()

    X = np.vstack(X)
    Y = np.vstack(Y)

    if mono:
        monoframes = np.nonzero(np.sum(Y,axis=1)==1)
        X = X[monoframes]
        Y = Y[monoframes]

    return np.vstack(X),np.vstack(Y)

def load_labels(handles, num_labels=128, mono=False):
    X = []
    Y = []
    conn = psycopg2.connect("dbname={} user={}".format(database,user))
    cur = conn.cursor()
    for id in handles.keys():
        print('+',end=" ")
        filename = handles[id]
        fs, record = wavfile.read(filename)
        assert fs == 44100
        X.append(record)
        
        cur.callproc('get_labels', ['labels',id])
        cur2 = conn.cursor('labels')
        labels = np.zeros(shape=(record.shape[0],num_labels))
        for label in cur2:
            start_time = int(float(label[0])*fs)
            end_time = int(float(label[1])*fs)
            note = label[2]
            for i in range(start_time,end_time):
                labels[i,note] = 1

        Y.append(labels)
                
        cur2.close()
    conn.close()

    X = np.vstack(X)
    Y = np.vstack(Y)

    if mono:
        monoframes = np.nonzero(np.sum(Y,axis=1)==1)
        X = X[monoframes]
        Y = Y[monoframes]

    # throw away unused labels
    Y = Y[:,36:68]

    # combine channels
    X = np.sum(X,axis=1)

    return X,Y

def sparse_labels(handles):
    labeled_data = dict()
    
    conn = psycopg2.connect("dbname={} user={}".format(database,user))
    cur = conn.cursor()
    for id in handles.keys():
        labels = IntervalTree()
        
        print('+',end=" ")
        filename = handles[id]
        fs, record = wavfile.read(filename)
        assert fs == 44100            # 44kHz
        assert len(record.shape) == 1 # mono
        X = record/32768.             # [-1,1)

        cur.callproc('get_labels', ['labels',id])
        cur2 = conn.cursor('labels')
        for label in cur2:
            start_time = int(float(label[0])*fs)
            end_time = int(float(label[1])*fs)
            note = label[2]
            labels[start_time:end_time] = note
        Y = labels

        labeled_data[str(id)] = (X,Y)
            
        cur2.close()
    conn.close()

    return labeled_data
