'''
@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def crossplot_pca(pca, labels):
    Xax = pca[:,0]
    Yax = pca[:,1]
    class_names = np.unique(labels)
    cdict={0:'red',1:'green'}
    labl={0:class_names[0],1:class_names[1]}
    marker={0:'*',1:'o'}
    alpha={0:.3, 1:.5}
    fig,ax=plt.subplots(figsize=(7,5))
    fig.patch.set_facecolor('white')
    for l in np.unique(labels):
        ix=np.where(labels==l)
        if l == class_names[0]:
            ax.scatter(Xax[ix],Yax[ix],c=cdict[0],s=40,
                    label=labl[0],marker=marker[0],alpha=alpha[0])
        else:
            ax.scatter(Xax[ix],Yax[ix],c=cdict[1],s=40,
                    label=labl[1],marker=marker[1],alpha=alpha[1])

    
    plt.xlabel("First Principal Component",fontsize=14)
    plt.ylabel("Second Principal Component",fontsize=14)
    plt.legend()
    plt.show()
    

def crossplot_dual_features(features, feature_names):
    fig,ax =plt.subplots(figsize=(10, 10))
       
    ax.scatter(features[feature_names[0]], features[feature_names[1]], s = 5, color='magenta', label='check', alpha=0.3)
    ax.set_xlabel(feature_names[0],fontsize=12)
    ax.set_ylabel(feature_names[1],fontsize=12)
    
    plt.tight_layout()
    plt.show()


def crossplot_features(features, class_names, display_type = 'histogram'):
     

    event = features[features['Class'] == 1]
    noise = features[features['Class'] == 2]

    events = event.drop(['Class','FileName','FeatureID', 'ClassLabels'],axis=1).as_matrix()
    noises = noise.drop(['Class','FileName','FeatureID', 'ClassLabels'],axis=1).as_matrix()

    dataset = features.drop(['Class','FileName','FeatureID','ClassLabels'],axis=1)
    #data = features.drop(['Class','FileName','FeatureID','ClassLabels'],axis=1).values
    feature_names = dataset.columns.values.tolist()
    feature_count = dataset.columns.size 

    if display_type == 'histogram':
        fig,axes =plt.subplots(10,5, figsize=(12, 9)) # 5 columns each containing 10 figures, total 50 features  
        ax=axes.ravel()# flat axes with numpy ravel
        for i in range(feature_count):
            #_,bins=np.histogram(data[:,i],bins=feature_count)
            n_bins = feature_count
            
            ax[i].hist(events[:,i],bins=n_bins,color='r',alpha=.5)# red color for malignant class
            ax[i].hist(noises[:,i],bins=n_bins,color='g',alpha=0.3)# alpha is           for transparency in the overlapped region 
            
            ax[i].set_title(feature_names[i],fontsize=9)
            ax[i].axes.get_xaxis().set_visible(False) # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
            ax[i].set_yticks(())
            ax[0].legend([class_names[0], class_names[1]],loc='best',fontsize=8)
        
    elif display_type == 'scatter':
        fig,axes =plt.subplots(10,5, figsize=(12, 9)) # 5 columns each containing 10 figures, total 50 features  

    plt.tight_layout()# let's make good plots
    plt.show()

def plot_coefficients(classifier, feature_names, top_features=20):
    from sklearn.feature_extraction.text import CountVectorizer
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()


def visualize_ml_result(data, label, classifier, class_name, step = 1, count = None):
    '''
    Only works for two features
    '''

    
    if count == None:        
        X_set, y_set = data, label
    elif  len(label) < count:
        X_set, y_set = data, label
    else:
        X_set, y_set = data[:50,:], label[:50]

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = step),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = step))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        print(i,j)
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('SVM (Training set)')
    plt.xlabel(class_name[0])
    plt.ylabel(class_name[1])
    plt.legend()
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()