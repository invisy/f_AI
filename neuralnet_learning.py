import os
import time
from libs.wav_nn_learn import *
import configs.config as cfg

epochsNumber = 50
neuralNetName = f'NN_L{epochsNumber}'

if input('Learn neuralnet? y/n\n') == 'y':
    if not os.path.isdir(cfg.learningResultPath):
        os.makedirs(cfg.learningResultPath)

    print('---start Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    Learn_NN_5L_(TrainDir=cfg.trainPath, 
                ValidDir=cfg.validPath,
                RezDir=cfg.learningResultPath,
                NN_Name=neuralNetName, Epochs=epochsNumber, window_size=25, windoe_fuction='hann')

    print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

if not os.path.isdir(cfg.testResultPath):
    os.makedirs(cfg.testResultPath)
    
if input('Test neuralnet? y/n\n') == 'y':
        TestNN_(NetName=os.path.join(cfg.learningResultPath, f'{neuralNetName}_Best.hdf5'),
                SourceDir=cfg.testPath,
                TargetFile=os.path.join(cfg.testResultPath, f'{neuralNetName}_rez_test'),
                window_size=25)

        TestNN_(NetName=os.path.join(cfg.learningResultPath, f'{neuralNetName}_Best.hdf5'),
                SourceDir=cfg.validPath,
                TargetFile=os.path.join(cfg.testResultPath, f'{neuralNetName}_rez_valid'),
                window_size=25)

        TestNN_(NetName=os.path.join(cfg.learningResultPath, f'{neuralNetName}_Best.hdf5'),
                SourceDir=cfg.trainPath,
                TargetFile=os.path.join(cfg.testResultPath, f'{neuralNetName}_rez_train'),
                window_size=25)

if input('Blind test? y/n\n') == 'y':
        if not os.path.isdir(cfg.blindTestResultPath):
                os.makedirs(cfg.blindTestResultPath)

        TestNN_(NetName=os.path.join(cfg.learningResultPath, f'{neuralNetName}_Best.hdf5'),
                SourceDir=cfg.blindtTestClassifiedPath,
                TargetFile=os.path.join(cfg.blindTestResultPath, f'{neuralNetName}_rez_blind_test'),
                window_size=25)