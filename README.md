# ml4ms
Machine Learning for Microseismic - A Python package for implementing automated microseismic event detection with machine learning.
This is also the companion repository for our paper titled "Using Supervised Machine Learning to Distinguish Microseismic from Noise Events".

## Classifier Showcase


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
'''
@incollection{zhao2017using,
  title={Using supervised machine learning to distinguish microseismic from noise events},
  author={Zhao, Zhengguang and Gross, Lutz},
  booktitle={SEG Technical Program Expanded Abstracts 2017},
  pages={2918--2923},
  year={2017},
  publisher={Society of Exploration Geophysicists}
}
```
