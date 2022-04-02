# Neuralnet sound recognition

## Table of contents
[Install dependencies](https://github.com/invisy/f_AI/new/main#install-dependencies)
<br>
[Prepare data](https://github.com/invisy/f_AI/new/main#install-dependencies)


## Install dependencies

### Install requested libraries
```
$ pip install -U numpy
```
```
$ pip install -U scipy
```
```
$ pip install -U scikit-learn
```
```
$ pip install -U pandas
```
```
$ pip install plaidml-keras==0.6.4
```
```
$ pip install pydub
```

### Downgrade h5py module to fix problems with older version of plaidml
```
$ pip uninstall h5py
```
```
$ pip install h5py==2.10.0
```

### Set up plaidml
```
$ plaidml-setup
```

### Install FFmpeg and set PATH variable
[Link](https://www.ffmpeg.org/download.html)

## Prepare data

### Data tree

<pre>
├───data
│   ├───blind_data
│   │   ├───classified_data
│   │   └───source_data
│   │       ├───word1
│   │       ├───word2
│   │       └───word3
│   ├───learning_data
│   │   ├───balanced_data
│   │   ├───classified_data
│   │   ├───source_data
│   │   │   ├───word1
│   │   │   ├───word2
│   │   │   ├───word3
│   │   │   └───_background_noise_
│   │   └───split_data
│   │       ├───test
│   │       ├───train
│   │       └───valid
│   └───neuralnets
│       ├───nn_NeuralNet1
│       │   └───learning_test
│       │   └───blind_test
</pre>

You must create folders data/learning_data/source_data at the same folder where the scripts are. Also you must create folders data/blind_data/source_data if you want to do blind test. Other folders will be created automatically.