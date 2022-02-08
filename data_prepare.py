from genericpath import isdir
from os import listdir
from os.path import isfile, join

from libs.wav_data_prep import *
import hashlib

dataPath = os.path.join(os.path.dirname(__file__), 'data')

sourcePath = os.path.join(dataPath, 'source_data')
classifiedPath = os.path.join(dataPath, 'classified_data')
balancedPath = os.path.join(dataPath, 'balanced_data')
learningPath = os.path.join(dataPath, 'learning_data')

trainPath = os.path.join(learningPath, 'train')
testPath = os.path.join(learningPath, 'test')
validPath = os.path.join(learningPath, 'valid')

firstWord = "tree"
secondWord = "two"

allDirectories = [dir for dir in listdir(sourcePath) if(isdir(join(sourcePath, dir)))]
othersExceptList = ["_background_noise_", firstWord, secondWord]

if not os.path.isdir(classifiedPath):
    os.makedirs(classifiedPath)

    ConvertTo_8K(SourceDir=os.path.join(sourcePath, firstWord), TargetDirectory=classifiedPath, 
                                        Prefics=hashlib.md5(firstWord.encode()).hexdigest(), ClassType='cl_1')
    print(f'Finished cl_1 --- word: {firstWord}!')

    ConvertTo_8K(SourceDir=os.path.join(sourcePath, secondWord), TargetDirectory=classifiedPath, 
                                        Prefics=hashlib.md5(secondWord.encode()).hexdigest(), ClassType='cl_2')
    print(f'Finished cl_2 --- word: {secondWord}!')

    for word in allDirectories:
        if word not in othersExceptList:
            ConvertTo_8K(SourceDir=os.path.join(sourcePath, word), TargetDirectory=classifiedPath, 
                                                Prefics=hashlib.md5(word.encode()).hexdigest(), ClassType='cl_3')
            print(f'Finished cl_3 --- word: {word}!')
    print('Finished cl_3 all!')

if not os.path.isdir(balancedPath):
    os.makedirs(balancedPath)
    Balance_Sample(classifiedPath, balancedPath)

if not os.path.isdir(learningPath):
    os.makedirs(trainPath)
    os.makedirs(testPath)
    os.makedirs(validPath)
    Divide_TrainTestValid(SourceDirectory=balancedPath,
                          TrainDirectory=trainPath,
                          TestDirectory=testPath, 
                          ValidDirectory=validPath, 
                          TestPercent=0.1,
                          ValidPercent=0.1)