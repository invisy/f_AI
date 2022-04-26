import os
from pathlib import Path

dataPath = os.path.join(Path(os.path.dirname(__file__)).parent.absolute(), 'data')
dataSetPath = os.path.join(dataPath, 'learning_data')
neuralnetsPath = os.path.join(dataPath, 'neuralnets')

learningPath = os.path.join(dataSetPath, 'split_data')
trainPath = os.path.join(learningPath, 'train')
testPath = os.path.join(learningPath, 'test')
validPath = os.path.join(learningPath, 'valid')

learningResultPath = os.path.join(neuralnetsPath, 'learning_results')
testResultPath = os.path.join(learningResultPath, 'learning_test')

calibrateLearningResultPath = os.path.join(neuralnetsPath, 'calibrate_results')
chartsPath = os.path.join(calibrateLearningResultPath, 'charts')

blindtTestClassifiedPath = os.path.join(dataPath, 'blind_data', 'classified_data')
blindTestResultPath = os.path.join(learningResultPath, 'blind_test')

blindDataPath = os.path.join(dataPath, 'blind_data')
blindSourceDataPath = os.path.join(blindDataPath, 'source_data')
blindClassifiedPath = os.path.join(blindDataPath, 'classified_data')