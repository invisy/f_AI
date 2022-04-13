import multiprocessing
from libs.wav_nn_learn_v2 import *
import matplotlib.pyplot as plt

dataPath = os.path.join(os.path.dirname(__file__), 'data')
dataSetPath = os.path.join(dataPath, 'learning_data')
neuralnetsPath = os.path.join(dataPath, 'neuralnets')
chartsPath = os.path.join(neuralnetsPath, 'charts')

learningPath = os.path.join(dataSetPath, 'split_data')
trainPath = os.path.join(learningPath, 'train')
testPath = os.path.join(learningPath, 'test')
validPath = os.path.join(learningPath, 'valid')
learningResultPath = os.path.join(neuralnetsPath, 'learning_results')
testResultPath = os.path.join(learningResultPath, 'learning_test')

blindtTestClassifiedPath = os.path.join(dataPath, 'blind_data', 'classified_data')
blindTestResultPath = os.path.join(learningResultPath, 'blind_test')

epochsNumber = 50
maxNeuronsInLayerNumber = 100
maxLayersNumber = 5


def multiprocessTrainLayer(previousLayersValues):
    manager = multiprocessing.Manager()
    layerResult = manager.dict()

    threads = list()

    #cpuCount = multiprocessing.cpu_count()
    cpuCount = 1
    neuronNumbersPerCoreCount = int(maxNeuronsInLayerNumber / cpuCount) + 1

    layerNumber = len(previousLayersValues)+1
    numbers = []

    max_neurons = maxNeuronsInLayerNumber+1
    if len(previousLayersValues) > 0:
        max_neurons = previousLayersValues[-1]+1

    for currentLayerNeurons in range(4, max_neurons):
        variant = previousLayersValues.copy()
        variant.append(currentLayerNeurons)
        numbers.append(variant)

    for cpuIndex in range(0, cpuCount):
        fromIndex = cpuIndex * neuronNumbersPerCoreCount
        toIndex = min((cpuIndex + 1) * neuronNumbersPerCoreCount, len(numbers))
        neuronNumbersPart = numbers[fromIndex:toIndex]
        t = multiprocessing.Process(target=trainLayer, args=(neuronNumbersPart, layerResult))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    plt.plot(layerResult.keys(), layerResult.values())
    plt.xlabel('layer neurons')
    plt.ylabel('validation accuracy')
    plt.savefig(f'{chartsPath}\\Result_{layerNumber}L.png', bbox_inches="tight")
    plt.clf()
    # plt.show()

    currentLayerNeuronNumber = FindMedianNeuronsNumber(layerResult)
    print (f'Best neuron number for {layerNumber} layer  is - {currentLayerNeuronNumber}', end='\n')

    return currentLayerNeuronNumber

def FindMedianNeuronsNumber(layerResult):
    sortedDict = sorted(layerResult.items(), key = lambda kv:(kv[1], kv[0]))
    elementNumber = int(len(sortedDict)/2)
    return sortedDict[elementNumber][0]


def trainLayer(numbers, layerResult):
    layerNumber = len(numbers[0])
    for neuronNumber in numbers:
        neuralNetName = f'NN_{epochsNumber}E_{layerNumber}L_{neuronNumber}'
        Learn_NN_5L_(TrainDir=trainPath,
                     ValidDir=validPath,
                     RezDir=learningResultPath,
                     NN_Name=neuralNetName,
                     neuronNumbersArray=neuronNumber,
                     Epochs=epochsNumber, window_size=25, windoe_fuction='hann')

        totalaccuracy = TestNN_(NetName=os.path.join(learningResultPath, f'{neuralNetName}_Best.hdf5'),
                                SourceDir=testPath,
                                TargetFile=os.path.join(learningResultPath, f'{neuralNetName}_rez_test'),
                                window_size=25)

        key = neuronNumber[-1]
        layerResult[key] = totalaccuracy

    return layerResult

#if input('Learn neuralnet? y/n\n') == 'y':
if __name__ == '__main__':
    if not os.path.isdir(learningResultPath):
        os.makedirs(learningResultPath)

    if not os.path.isdir(chartsPath):
        os.makedirs(chartsPath)

    print('---start Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), end='\n')

    bestModelNeuronsArray = []
    for layerNumber in range(1, maxLayersNumber + 1):
        bestModelNeuronsArray.append(multiprocessTrainLayer(bestModelNeuronsArray))

    print(f'The best model - {bestModelNeuronsArray}', end='\n')
    print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), end='\n')



"""
        numbers = []
        getNeuronNumbers(arr, len(arr), layerNumber, numbers)
        layerResult = dict()
        for neuronNumbers in numbers:
            neuralNetName = f'NN_{epochsNumber}E_{layerNumber}L_{neuronNumbers}'
            max_val_acc_index = Learn_NN_5L_(TrainDir=trainPath,
                                             ValidDir=validPath,
                                             RezDir=learningResultPath,
                                             NN_Name=neuralNetName,
                                             neuronNumbersArray=neuronNumbers,
                                             Epochs=epochsNumber, window_size=25, windoe_fuction='hann')
            key = '\n'.join(str(x) for x in neuronNumbers)
            layerResult[key] = max_val_acc_index

        plt.plot(layerResult.keys(), [i[0] for i in layerResult.values()])
        plt.xlabel('layer neurons')
        plt.ylabel('validation accuracy')
        plt.savefig(f'{chartsPath}\\Result_{layerNumber}L.png', bbox_inches="tight")
        plt.clf()
        # plt.show()

    print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
"""
""" # temporary disable
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
"""