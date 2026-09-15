# genrenet
The ultimate music genre classifier using both acoustic features and spectrogram

## Dataset
The dataset is a collection of 30 seconds music samples in four different genres and is gathered from [deezer.com](https://www.deezer.com/). Audio preview samples were obtained through the Deezer API and used solely for feature extraction. The original audio recordings are not redistributed. 
The dataset including only extracted features is available at: https://www.kaggle.com/datasets/kianeliasi/genrenet 

## Features
Mean and variance of the follwing features:
* Root Mean Square
* Chroma Feature
* Spectral Centroid
* Spectral Bandwidth
* Rolloff Frequency
* Zero-Crossing Rate
* Mel-Frequency Cepstrum Coefficients

Additionally:
+ Black and white pectrogram images

## Models
* Fully-connected deep network for acoustic features
* Convolutional Neural Network for spectrograms

Aggregation method: choose the model with more confident outputs
