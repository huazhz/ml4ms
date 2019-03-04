'''
@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_predictions(result, trace, wins):
    fig,ax = plt.subplots(1,1)
    ax.set_xlabel('Samples') ; ax.set_ylabel('Normalized Amplitude')
    ax.set_xlim(0,30000) 
    ax.set_ylim(-1.0,1.0)
    
    #ax.plot(trace)

    for ind, val in enumerate(wins):
        if result[ind] == 1:
            ax.plot(val, trace[val], c = 'r' )
        else:
            ax.plot(val, trace[val], c = 'b' )


    plt.show()
def plot_correlations(features, feature_names):

    import seaborn as sns
    s=sns.heatmap(features[feature_names].corr(),cmap='coolwarm') 
    s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
    s.set_xticklabels(s.get_xticklabels(),rotation=65,fontsize=7)
    plt.tight_layout()
    plt.show()

def heatplot_pca(pca, feature_names):
    '''
    Designed for 3 components
    These principal components are calculated only from features and no information from classes are considered. 
    So PCA is unsupervised method and it’s difficult to interpret the two axes as they are some complex mixture 
    of the original features.
    We can make a heat-plot to see how the features mixed up to create the components.
    '''
    
    plt.matshow(pca.components_,cmap='viridis')
    plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
    plt.colorbar()
    plt.xticks(range(len(feature_names)), feature_names,rotation=65,ha='left')
    plt.tight_layout()
    plt.show()#

def crossplot_pca(X_pca, labels):
    Xax = X_pca[:,0]
    Yax = X_pca[:,1]
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

    events = event.drop(['Class','FileName','FeatureID', 'ClassLabels'],axis=1).values
    noises = noise.drop(['Class','FileName','FeatureID', 'ClassLabels'],axis=1).values

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


def visualize_ml_result(classifier_name, features, feature_names, class_names, step = 0.01, count = 50):
    '''
    Only works for two features
    '''

    if len(feature_names) != 2:
        print('Error!')
        return

    if classifier_name=='SVM': 

        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)       
        
    
    elif classifier_name=='KNN': 

        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

    elif classifier_name=='DecisionTree': 

        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

    elif classifier_name=='LogisticRegression': 

        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
    
    elif classifier_name=='LDA': 

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        classifier = LinearDiscriminantAnalysis()

    elif classifier_name=='QDA': 

        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        classifier = QuadraticDiscriminantAnalysis()

    data = features.loc[:, feature_names].values
    label = features['Class'].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier.fit(X_train, y_train)

    if count != None and len(label) > count:        
        X_train_set = X_train[:count,:].copy()
        y_train_set = y_train[:count].copy()
        X_test_set = X_test[:count,:].copy()
        y_test_set = y_test[:count].copy()
    
    else:
        X_train_set = X_train.copy()
        y_train_set = y_train.copy()
        X_test_set = X_test.copy()
        y_test_set = y_test.copy()
    
    plt.figure(1)

    for n in range(2):

        if n == 0:
            set_name = '(Training Set)'            
            X = X_train_set
            y = y_train_set
        else:
            set_name = '(Test Set)'            
            X = X_test_set
            y = y_test_set

        plt.subplot(1,2, n+1)

        X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1, stop = X[:, 0].max() + 1, step = step),
                            np.arange(start = X[:, 1].min() - 1, stop = X[:, 1].max() + 1, step = step))
        # plt.contourf(X1, X2, classifier.clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        #             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.contourf(X1, X2, classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape),
                    alpha = 0.75, cmap = ListedColormap(('red', 'green'))) # alternative code

        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y)):
            print(i,j)
            plt.scatter(X[y == j, 0], X[y == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title(classifier_name+' Classification '+set_name)
        plt.xlabel(class_names[0])
        plt.ylabel(class_names[1])
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


def compare_classifiers(dataset):

    '''
    # Code source: Gaël Varoquaux
    #              Andreas Müller
    # Modified for documentation by Jaques Grobler
    # License: BSD 3 clause

    # Original codes can be found here: https://scikit-learn.org/stable/auto_examples/
            # classification/plot_classifier_comparison.html#
            # sphx-glr-auto-examples-classification-plot-classifier-comparison-py

    '''


    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    h = .02  # step size in the mesh

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
            "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(dataset):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(dataset), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(dataset), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                    edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.show()