import multiprocessing
from libs.wav_nn_learn_calibrate import *
import matplotlib.pyplot as plt
import configs.config as cfg

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
    plt.savefig(f'{cfg.chartsPath}\\Result_{layerNumber}L.png', bbox_inches="tight")
    plt.clf()

    currentLayerNeuronNumber = FindBestNeuronsNumber(layerResult)
    print (f'Best neuron number for {layerNumber} layer  is - {currentLayerNeuronNumber}', end='\n')

    return currentLayerNeuronNumber

def FindBestNeuronsNumber(layerResult):
    window = 5
    layerResultList = sorted(layerResult.items(), key = lambda kv:(kv[0]))
    
    if len(layerResultList) > 5:
        bestWindow = {"index": 0, "value": 0} 
        
        for i in range(0, len(layerResultList)-(window-1)):
            subLayerResultList = layerResultList[i:i+window]
            windowSumm = sum([v[1] for v in subLayerResultList])
            if (windowSumm > bestWindow["value"]):
                bestWindow["value"] = windowSumm
                bestWindow["index"] = layerResultList[i+(int(window/2))][0]
                
        return bestWindow["index"]
    else:
        bestValue = {"index": 0, "value": 0} 
        for layerResult in layerResultList:
            if(layerResult[1] > bestValue["value"]):
                bestValue["value"] = layerResult[1]
                bestValue["index"] = layerResult[0]
                
        return bestValue["index"]

def trainLayer(numbers, layerResult):
    layerNumber = len(numbers[0])
    for neuronNumber in numbers:
        neuralNetName = f'NN_{epochsNumber}E_{layerNumber}L_{neuronNumber}'
        if(not os.path.isfile(os.path.join(cfg.calibrateLearningResultPath, f'{neuralNetName}_Final.hdf5'))):
            Learn_NN_5L_(TrainDir=cfg.trainPath,
                        ValidDir=cfg.validPath,
                        RezDir=cfg.calibrateLearningResultPath,
                        NN_Name=neuralNetName,
                        neuronNumbersArray=neuronNumber,
                        Epochs=epochsNumber, window_size=25, windoe_fuction='hann', bestOnly=False)

        totalaccuracy = TestNN_(NetName=os.path.join(cfg.calibrateLearningResultPath, f'{neuralNetName}_Final.hdf5'),
                                SourceDir=cfg.testPath,
                                TargetFile=os.path.join(cfg.calibrateLearningResultPath, f'{neuralNetName}_rez_test'),
                                window_size=25)

        key = neuronNumber[-1]
        layerResult[key] = totalaccuracy

    return layerResult

if __name__ == '__main__':
    print(cfg.calibrateLearningResultPath)
    if not os.path.isdir(cfg.calibrateLearningResultPath):
        os.makedirs(cfg.calibrateLearningResultPath)

    if not os.path.isdir(cfg.chartsPath):
        os.makedirs(cfg.chartsPath)

    print('---start Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), end='\n')

    bestModelNeuronsArray = []
    for layerNumber in range(1, maxLayersNumber + 1):
        if len(bestModelNeuronsArray) > 0 and bestModelNeuronsArray[-1] < 6:
            break
        bestModelNeuronsArray.append(multiprocessTrainLayer(bestModelNeuronsArray))

    print(f'The best model - {bestModelNeuronsArray}', end='\n')
    print('---end  Learn---', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), end='\n')
