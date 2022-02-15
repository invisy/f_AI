from genericpath import isdir
from os import listdir
from os.path import isfile, join

from libs.wav_data_prep import *
import hashlib
import shutil

dataPath = os.path.join(os.path.dirname(__file__), 'data')

def ClassifySounds(sourcePath, classifiedPath, regenerate=False):
    if not os.path.isdir(classifiedPath) or regenerate:
        shutil.rmtree(classifiedPath, ignore_errors=True)
        os.makedirs(classifiedPath)

        firstWord = "tree"
        secondWord = "two"

        allDirectories = [dir for dir in listdir(sourcePath) if(isdir(join(sourcePath, dir)))]
        othersExceptList = ["_background_noise_", firstWord, secondWord]

        ConvertToWavFromFolder(SourceDir=os.path.join(sourcePath, firstWord), TargetDirectory=classifiedPath, 
                                            Prefics=hashlib.md5(firstWord.encode()).hexdigest(), ClassType='cl_1')
        print(f'Finished classifying of cl_1 --- word: {firstWord}!')

        ConvertToWavFromFolder(SourceDir=os.path.join(sourcePath, secondWord), TargetDirectory=classifiedPath, 
                                            Prefics=hashlib.md5(secondWord.encode()).hexdigest(), ClassType='cl_2')
        print(f'Finished classifying of cl_2 --- word: {secondWord}!')

        for word in allDirectories:
            if word not in othersExceptList:
                ConvertToWavFromFolder(SourceDir=os.path.join(sourcePath, word), TargetDirectory=classifiedPath, 
                                                    Prefics=hashlib.md5(word.encode()).hexdigest(), ClassType='cl_3')
                print(f'Finished classifying of cl_3 --- word: {word}!')
        print('Finished classifying of cl_3 --- all files!')

def BalanceClassifiedSounds(classifiedPath, balancedPath, regenerate=False):
    if os.path.isdir(balancedPath) and (not os.path.isdir(balancedPath) or regenerate):
        shutil.rmtree(balancedPath, ignore_errors=True)
        os.makedirs(balancedPath)
        Balance_Sample(classifiedPath, balancedPath)

def SplitBalancedSounds(balancedPath, splittedPath, trainPath, testPath, validPath, regenerate=False):
    if os.path.isdir(balancedPath) and (not os.path.isdir(splittedPath) or regenerate):
        shutil.rmtree(splittedPath, ignore_errors=True)
        os.makedirs(trainPath)
        os.makedirs(testPath)
        os.makedirs(validPath)
        Divide_TrainTestValid(SourceDirectory=balancedPath,
                            TrainDirectory=trainPath,
                            TestDirectory=testPath, 
                            ValidDirectory=validPath, 
                            TestPercent=0.1,
                            ValidPercent=0.1)

def PrepareLearningData(regenerate = False):
    dataSetPath = os.path.join(dataPath, 'learning_data')

    sourceDataPath = os.path.join(dataSetPath, 'source_data')
    classifiedPath = os.path.join(dataSetPath, 'classified_data')
    balancedPath = os.path.join(dataSetPath, 'balanced_data')
    splitDataPath = os.path.join(dataSetPath, 'split_data')

    trainPath = os.path.join(splitDataPath, 'train')
    testPath = os.path.join(splitDataPath, 'test')
    validPath = os.path.join(splitDataPath, 'valid')

    ClassifySounds(sourceDataPath, classifiedPath, regenerate)
    BalanceClassifiedSounds(classifiedPath, balancedPath, regenerate)
    SplitBalancedSounds(balancedPath, splitDataPath, trainPath, testPath, validPath, regenerate)

def PrepareBlindTestData(regenerate = False): 
    blindDataPath = os.path.join(dataPath, 'blind_data')
    blindSourceDataPath = os.path.join(blindDataPath, 'source_data')
    classifiedPath = os.path.join(blindDataPath, 'classified_data')

    ClassifySounds(blindSourceDataPath, classifiedPath, regenerate)

#-----------------------------------------------#

if __name__ == '__main__':
    if input('Regenerate learning data? y/n\n') == 'y':
        PrepareLearningData(True)
    if input('Regenerate blind test data? y/n\n') == 'y':
        PrepareBlindTestData(True)