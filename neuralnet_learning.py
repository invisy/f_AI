import os
import time
from libs.wav_nn_learn import *

dataPath = os.path.join(os.path.dirname(__file__), 'data')
dataSetPath = os.path.join(dataPath, 'learning_data')
neuralnetsPath = os.path.join(dataPath, 'neuralnets')

learningPath = os.path.join(dataSetPath, 'split_data')
trainPath = os.path.join(learningPath, 'train')
testPath = os.path.join(learningPath, 'test')
validPath = os.path.join(learningPath, 'valid')
learningResultPath = os.path.join(neuralnetsPath, 'nn_autoEpochsNumber')
testResultPath = os.path.join(learningResultPath, 'learning_test')

blindtTestClassifiedPath = os.path.join(dataPath, 'blind_data', 'classified_data')
blindTestResultPath = os.path.join(learningResultPath, 'blind_test')

epochsNumber = 999
neuralNetName = f'NN_L{epochsNumber}'

if input('Learn neuralnet? y/n\n') == 'y':
    if not os.path.isdir(learningResultPath):
        os.makedirs(learningResultPath)

    print('---start Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    Learn_NN_5L_(TrainDir=trainPath, 
                ValidDir=validPath,
                RezDir=learningResultPath,
                NN_Name=neuralNetName, Epochs=epochsNumber, window_size=25, windoe_fuction='hann')

    print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

if not os.path.isdir(testResultPath):
    os.makedirs(testResultPath)
    
if input('Test neuralnet? y/n\n') == 'y':
        TestNN_(NetName=os.path.join(learningResultPath, f'{neuralNetName}_Best.hdf5'),
                SourceDir=testPath,
                TargetFile=os.path.join(testResultPath, f'{neuralNetName}_rez_test'),
                window_size=25)

        TestNN_(NetName=os.path.join(learningResultPath, f'{neuralNetName}_Best.hdf5'),
                SourceDir=validPath,
                TargetFile=os.path.join(testResultPath, f'{neuralNetName}_rez_valid'),
                window_size=25)

        TestNN_(NetName=os.path.join(learningResultPath, f'{neuralNetName}_Best.hdf5'),
                SourceDir=trainPath,
                TargetFile=os.path.join(testResultPath, f'{neuralNetName}_rez_train'),
                window_size=25)

if input('Blind test? y/n\n') == 'y':
        TestNN_(NetName=os.path.join(learningResultPath, f'{neuralNetName}_Best.hdf5'),
                SourceDir=blindtTestClassifiedPath,
                TargetFile=os.path.join(blindTestResultPath, f'{neuralNetName}_rez_blind_test'),
                window_size=25)