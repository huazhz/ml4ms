# ml4ms
Machine Learning for Microseismic - A Python package for implementing automated microseismic event detection with machine learning.
This is also the companion repository for our paper titled ["Using Supervised Machine Learning to Distinguish Microseismic from Noise Events"](https://www.researchgate.net/publication/319622797_Using_Supervised_Machine_Learning_to_Distinguish_Microseismic_from_Noise_Events?_sg=XCg5ScjsVBCwpHsV6MaoNuW8Et7dcLb7PdcH3tcYW7Cm1NGh9HVbuS3Juh1-XadkFXs91zDJnBYFlK7jjqRkvUQAfhIwN3VmiQWQgvpN.fsWiGLMtbYWUL-7t-BlYfurJ1KPUnffA7IXytqI7qARTbwHbKCzKX7eNAf7TDSUBDRdpgvKTbMrkqTLaxVjS9Q).

This package was translated from previous Matlab codes. As Python is more popular in machine learning and deep learning, we decided to use Python for future coding.

## Classifier Showcase (Matlab version)
<div align=center>![SVM Classifier](https://github.com/uqzzhao/ml4ms/blob/master/examples/images/SVM_0.9_0817_prediction.gif)


## Data
The data used in this project comes from various sources. Users should prepare their own data for training purpose.

## Code
The code is divided as follows:

The ml4ms.py python file contains the necessary code to run an experiement.

The utils folder contains the necessary functions to read the datasets and visualize the plots.


## Prerequisites
All python packages needed are listed below and can be installed simply using the pip command.
* [numpy](http://www.numpy.org/)  
* [pandas](https://pandas.pydata.org/)  
* [sklearn](http://scikit-learn.org/stable/)  
* [scipy](https://www.scipy.org/)  
* [matplotlib](https://matplotlib.org/)  
* [seaborn](https://seaborn.pydata.org/)


## Results
Our results showed that a SVM classifier with Gaussian kernel performs best for the time series (microseismic and noise events) classification task.


## Reference
If you re-use this work, please cite:

Zhao, Z., & Gross, L. (2017). Using supervised machine learning to distinguish microseismic from noise events. In SEG Technical Program Expanded Abstracts 2017 (pp. 2918-2923). Society of Exploration Geophysicists.

or

```
@incollection{zhao2017using,
  title={Using supervised machine learning to distinguish microseismic from noise events},
  author={Zhao, Zhengguang and Gross, Lutz},
  booktitle={SEG Technical Program Expanded Abstracts 2017},
  pages={2918--2923},
  year={2017},
  publisher={Society of Exploration Geophysicists}
}
```
