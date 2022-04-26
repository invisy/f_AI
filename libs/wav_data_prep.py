import numpy as np
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import LabelBinarizer
labelbinarizer = LabelBinarizer()
import os
from scipy import signal
from shutil import copyfile
import random
import multiprocessing
import math
from pydub import AudioSegment, exceptions
from pydub.exceptions import CouldntDecodeError
import configs.config as cfg

CLASSIFIED_SAMPLE_RATE = 8000
CLASSIFIED_SAMPLE_WIDTH = 2

def Convert_To_06(samples):
    sample_rate = CLASSIFIED_SAMPLE_RATE
    WindowSize = 40
    LeftSeek = 300
    RezFileLength = 0.6

    if len(samples) <= int(0.6*sample_rate):
        return samples

    Samples_SUM = np.arange(len(samples), dtype='float64')
    ResultS = np.arange(int(RezFileLength * sample_rate), dtype=samples.dtype)
    Samples_Float = np.arange(len(samples), dtype='float64')

    # ------ARRAY -----AverageSum--------------------------
    Average = 0.0
    for i in range(len(samples)):
        Average += samples[i]

    Average = Average / len(samples)
    for i in range(len(samples)):
        Samples_Float[i] = abs(samples[i] - Average)

    MaxSum = 0
    TotalSum = 0
    for i in range(len(Samples_SUM)):
        Samples_SUM[i] = 0
    i = WindowSize
    while i < (len(samples) - WindowSize - 1):
        CurentSum = 0
        J = -WindowSize
        while J < WindowSize:
            CurentSum = CurentSum + Samples_Float[i + J]
            J = J + 1
        Samples_SUM[i] = CurentSum / (2 * WindowSize + 1)
        if MaxSum < Samples_SUM[i]:
            MaxSum = Samples_SUM[i]

        i += 1
    Thresholt = MaxSum / 6
    for asum in Samples_SUM:
        if Thresholt < asum:
            TotalSum += asum
    CenterIndex = 0
    CurentSum = Samples_SUM[0]

    while CurentSum < TotalSum / 2:
        CenterIndex += 1
        CurentSum += Samples_SUM[CenterIndex]

    StartIndex = 0
    BitRateCount = 0
    for i in range(len(Samples_SUM)):
        if Samples_SUM[i] > Thresholt:
            BitRateCount += 1
        else:
            BitRateCount = 0
        if BitRateCount > WindowSize * 4:
            if i > WindowSize * 4 + LeftSeek:
                StartIndex = int(i - (WindowSize * 4 + LeftSeek))
            break


    if CenterIndex < int(StartIndex + (sample_rate * RezFileLength / 2)) and (sample_rate * RezFileLength / 2) < CenterIndex:
        StartIndex = CenterIndex - (sample_rate * RezFileLength / 2)
    if StartIndex + sample_rate * RezFileLength > len(samples):
        StartIndex = len(samples) - sample_rate * RezFileLength
    StartIndex = int(StartIndex)
    for i in range(int(RezFileLength * sample_rate)):
        ResultS[i] = samples[i + StartIndex]
    return ResultS

def ConvertToWav(SourceDir, TargetDirectory, Prefics, ClassType, SoundFileName):
    #  -------Convert, Raname and Save File----------------------------------------

    soundFilePath = os.path.join(SourceDir, SoundFileName)
    try:
        soundFile = AudioSegment.from_file(soundFilePath)
        sampleWidth = soundFile.sample_width
        sampleRate = soundFile.frame_rate
        sampleChannelsNumber = soundFile.channels

        if sampleWidth != CLASSIFIED_SAMPLE_WIDTH:
            soundFile = soundFile.set_sample_width(2)
        if sampleRate != CLASSIFIED_SAMPLE_RATE:
            soundFile = soundFile.set_frame_rate(CLASSIFIED_SAMPLE_RATE)
        if sampleChannelsNumber != 1:
            soundFile = soundFile.set_channels(1)

        samples = soundFile.get_array_of_samples()
        numpySamples = np.array(samples)
        Samples_final = Convert_To_06(numpySamples)             #TODO rewrite function

        with open(os.path.join(cfg.blindSourceDataPath, "files_crypted.txt"), "a") as filesCryptedTxt:
            filesCryptedTxt.write(f'{SoundFileName} : {Prefics}_{ClassType}.wav\n')
        destinationPath = os.path.join(TargetDirectory, f'{Prefics}_{ClassType}.wav')

        wavfile.write(destinationPath, CLASSIFIED_SAMPLE_RATE, Samples_final)
    except CouldntDecodeError:
        print(f'File {SoundFileName} can`t be converted -- skipping!')

def ConvertToWavRange(SourceDir, TargetDirectory, Prefics, ClassType, wavFiles, fromIndex, toIndex):
    for fileIndex in range(fromIndex, toIndex):
        ConvertToWav(SourceDir, TargetDirectory, f'{Prefics}-{fileIndex}', ClassType, wavFiles[fileIndex])

def ConvertToWavFromFolder(SourceDir, TargetDirectory, Prefics, ClassType):
    #  -------Load Files----------------------------------------
    soundFiles = []
    for d, dirs, files in os.walk(SourceDir):
        for file in files:
            soundFiles.append(file)
    #  -------Convert, Raname and Save Files----------------------------------------
    threads = list()
    cpuCount = multiprocessing.cpu_count()
    soundFilesCount = len(soundFiles)

    if(soundFilesCount == 0):
        return

    if soundFilesCount < cpuCount:
        cpuCount = 1

    soundFilesPerCoreCount = math.ceil(soundFilesCount/cpuCount)

    for cpuIndex in range(0, cpuCount):
        fromIndex = min(cpuIndex*soundFilesPerCoreCount, soundFilesCount)
        toIndex = min((cpuIndex+1)*soundFilesPerCoreCount, soundFilesCount)
        if(fromIndex != toIndex):
            t = multiprocessing.Process(target=ConvertToWavRange,
                                        args = (SourceDir, TargetDirectory, Prefics, ClassType, soundFiles, fromIndex, toIndex))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
        
def Balance_Sample(SourceDirectory, BalancedSampleDirectory):
    Class1_Files = []
    Class2_Files = []
    Class3_Files = []
    Balanced_Files = []
    Class1_ID = "cl_1"
    Class2_ID = "cl_2"
    Class3_ID = "cl_3"

    Wav_Files = []
    for d, dirs, files in os.walk(SourceDirectory):
        for file in files:
            if file.endswith(".wav"):
                Wav_Files.append(file)
    for fwav in Wav_Files:
        if Class1_ID in fwav:
            Class1_Files.append(fwav)
        if Class2_ID in fwav:
            Class2_Files.append(fwav)
        if Class3_ID in fwav:
            Class3_Files.append(fwav)

    Class1_Length = len(Class1_Files)
    Class2_Length = len(Class2_Files)
    Class3_Length = len(Class3_Files)

    MinimalElementsCount = min(Class1_Length, Class2_Length, Class3_Length)

    for i in range(MinimalElementsCount):
        Class1_Length = len(Class1_Files)
        Class2_Length = len(Class2_Files)
        Class3_Length = len(Class3_Files)

        Index = random.randint(0, Class1_Length - 1)
        Balanced_Files.append(Class1_Files[Index])
        del Class1_Files[Index]
        Index = random.randint(0, Class2_Length - 1)
        Balanced_Files.append(Class2_Files[Index])
        del Class2_Files[Index]
        Index = random.randint(0, Class3_Length - 1)
        Balanced_Files.append(Class3_Files[Index])
        del Class3_Files[Index]

    for tFile in Balanced_Files:
        src = os.path.join(SourceDirectory, tFile)
        dst = os.path.join(BalancedSampleDirectory, tFile)
        copyfile(src, dst)

def Divide_TrainTestValid(SourceDirectory, TrainDirectory, TestDirectory, ValidDirectory, TestPercent, ValidPercent):
    Clas1_Files = []
    Clas2_Files = []
    Clas3_Files = []
    Test_Files = []
    Clas1_ID = "cl_1"
    Clas2_ID = "cl_2"
    Clas3_ID = "cl_3"

    Wav_Files = []
    for d, dirs, files in os.walk(SourceDirectory):
        for file in files:
            if file.endswith(".wav"):
                Wav_Files.append(file)
    for fwav in Wav_Files:
        if Clas1_ID in fwav:
            Clas1_Files.append(fwav)
        if Clas2_ID in fwav:
            Clas2_Files.append(fwav)
        if Clas3_ID in fwav:
            Clas3_Files.append(fwav)
  
    TestSize = len(Clas1_Files)
    if TestSize > len(Clas2_Files):
        TestSize = len(Clas2_Files)
    if TestSize > len(Clas3_Files):
        TestSize = len(Clas3_Files)
    ValidSize = int(TestSize * ValidPercent)
    TestSize = int(TestSize * TestPercent)

    # ------------------TEST----------------------------------------

    for i in range(TestSize):
        Index = random.randint(0, len(Clas1_Files) - 1)
        Test_Files.append(Clas1_Files[Index])
        del Clas1_Files[Index]
        Index = random.randint(0, len(Clas2_Files) - 1)
        Test_Files.append(Clas2_Files[Index])
        del Clas2_Files[Index]
        Index = random.randint(0, len(Clas3_Files) - 1)
        Test_Files.append(Clas3_Files[Index])
        del Clas3_Files[Index]

    for tFile in Test_Files:
        src = os.path.join(SourceDirectory, tFile)
        dst = os.path.join(TestDirectory, tFile)
        copyfile(src, dst)
   # ------------------TEST---------------------------------------


  # ------------------Valid----------------------------------------
    Test_Files = []
    for i in range(ValidSize):
        Index = random.randint(0, len(Clas1_Files) - 1)
        Test_Files.append(Clas1_Files[Index])
        del Clas1_Files[Index]
        Index = random.randint(0, len(Clas2_Files) - 1)
        Test_Files.append(Clas2_Files[Index])
        del Clas2_Files[Index]
        Index = random.randint(0, len(Clas3_Files) - 1)
        Test_Files.append(Clas3_Files[Index])
        del Clas3_Files[Index]

    for tFile in Test_Files:
        src = os.path.join(SourceDirectory, tFile)
        dst = os.path.join(ValidDirectory, tFile)
        copyfile(src, dst)
    # ------------------Valid---------------------------------------


    Train_Files = Clas1_Files + Clas2_Files + Clas3_Files
    for tFile in Train_Files:
        src = os.path.join(SourceDirectory, tFile)
        dst = os.path.join(TrainDirectory, tFile)
        copyfile(src, dst)
