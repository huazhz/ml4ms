'''
@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

'''

import os
import numpy as np


def wave_norm(dataset):    
    l = len(dataset)
    for i in np.arange(l):
        w = dataset[i, :]
        wnorm = w / max(abs(w))
        dataset[i, :] = wnorm
    return dataset

def segment_trace(trace, window_length, overlap_length, norm_flag = 1, outpath = None):   

    import pandas as pd 
   
    
    signal_length = len(trace)       
    step_length = window_length - overlap_length
    number_of_windows = int(np.floor((signal_length-window_length)/step_length) + 1)
    #data = trace.reshape((-1,1), order = 'F').copy()
    
    segments = np.ones((number_of_windows, window_length), dtype = np.float32)
    wins = []
    label_list = []
    file_name_list = []
        
    for i in range(number_of_windows):
        wstart = i * step_length
        wend = i * step_length + window_length

        label_list.append(0)
        file_name_list.append(str(wstart)+'_'+str(wend))

        segments[i,:] = trace[wstart:wend]
        if i == 0:
            wins = [np.arange(wstart, wend)]
        
        else:
            wins = np.append(wins, [np.arange(wstart, wend)], axis = 0)

    n_win = len(wins)

    # Normalization
    if norm_flag == 1:
        data = wave_norm(segments) # rescaling real valued numeric attributes into the range -1 and 1.
    else:
        data = segments

    labels = np.array((label_list)).reshape(-1, 1)    
    fnames = np.array((file_name_list)).reshape(-1, 1)    
    dataset = np.hstack((labels,data))

    dataset_with_fname = np.hstack((fnames,labels,data))   

    if outpath != None:
        pd.DataFrame(dataset_with_fname).to_csv(outpath)

    return dataset_with_fname, wins
          
     
    


def create_directory(directory_path): 

    if os.path.exists(directory_path): 

        return None

    else: 

        try: 

            os.makedirs(directory_path)

        except: 

            # in case another machine created the path meanwhile !:(

            return None 

        return directory_path

def read_dataset(root_dir,archive_name,dataset_name):

    datasets_dict = {}

    
    file_name = os.path.join(root_dir,'archives', archive_name, dataset_name, dataset_name)

    x_train, y_train = readucr(file_name+'_TRAIN.txt')

    x_test, y_test = readucr(file_name+'_TEST.txt')

    datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),

        y_test.copy())



    return datasets_dict


def readucr(filename):

    data = np.loadtxt(filename, delimiter = ',')

    Y = data[:,0]

    X = data[:,1:]

    return X, Y






def read_data(file_path):
    import pandas as pd

    column_names = ['user-id', 'activity',
                    'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data


def load_data(file_path):
    f = open(file_path)
    data = np.loadtxt(fname=f, delimiter=',')
    f.close()
    return data


def load_csv(file_path, header = 0, column_names = None):
    import pandas as pd
    f = open(file_path)
    data = pd.read_csv(file_path, header= header, names=column_names)
    f.close()
    return data


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=1)
    sigma = np.std(dataset, axis=1)
    return (dataset - mu) / sigma


def wave_norm(dataset):

    l = len(dataset)
    for i in np.arange(l):
        w = dataset[i, :]
        wnorm = w / max(abs(w))
        dataset[i, :] = wnorm
    return dataset




def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size / 2)


def segment_signal(data, window_size=90):
    from scipy import stats

    segments = np.zeros((0, window_size, 3))
    labels = np.zeros((0))

    for (start, end) in windows(data["timestamp"], window_size):
        start = int(start)
        # print(start)
        end = int(end)
        # print(end)
        x = data["x-axis"][start: end]
        # print(x)
        y = data["y-axis"][start: end]
        z = data["z-axis"][start: end]
        if(len(data["timestamp"][start: end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(
                data["activity"][start: end])[0][0])
    return segments, labels



def display_cm(cm, labels, hide_zeros=False,
                             display_metrics=False):
    """Display confusion matrix with labels, along with
       metrics such as Recall, Precision and F1 score.
       Based on Zach Guo's print_cm gist at
       https://gist.github.com/zachguo/10296432
    """

    precision = np.diagonal(cm)/cm.sum(axis=0).astype('float')
    recall = np.diagonal(cm)/cm.sum(axis=1).astype('float')
    F1 = 2 * (precision * recall) / (precision + recall)
    
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    F1[np.isnan(F1)] = 0
    
    total_precision = np.sum(precision * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_recall = np.sum(recall * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_F1 = np.sum(F1 * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    #print total_precision
    
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + " Pred", end=' ')
    for label in labels: 
        print("%{0}s".format(columnwidth) % label, end=' ')
    print("%{0}s".format(columnwidth) % 'Total')
    print("    " + " True")
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=' ')
        for j in range(len(labels)): 
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeros:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            print(cell, end=' ')
        print("%{0}d".format(columnwidth) % sum(cm[i,:]))
        
    if display_metrics:
        print()
        print("Precision", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % precision[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_precision)
        print("   Recall", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % recall[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_recall)
        print("       F1", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % F1[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_F1)
    
                  
def display_adj_cm(
        cm, labels, adjacent_facies, hide_zeros=False, 
        display_metrics=False):
    """This function displays a confusion matrix that counts 
       adjacent facies as correct.
    """
    adj_cm = np.copy(cm)
    
    for i in np.arange(0,cm.shape[0]):
        for j in adjacent_facies[i]:
            adj_cm[i][i] += adj_cm[i][j]
            adj_cm[i][j] = 0.0
        
    display_cm(adj_cm, labels, hide_zeros, 
                             display_metrics)



