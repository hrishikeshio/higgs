
import random,string,math,csv,pandas
import numpy as np
import matplotlib.pyplot as plt

def DataToArff(xs,labels,weights,header,title,fileName):
    outFile = open(fileName + ".arff","w")
    outFile.write("@RELATION " + title + "\n\n")
    #header
    for feature in header:
        outFile.write("@ATTRIBUTE " + feature + 
                      " NUMERIC\n")
    # multiboost requiresw eight in two columns
    outFile.write("@ATTRIBUTE classSignal NUMERIC\n")
    outFile.write("@ATTRIBUTE classBackground NUMERIC\n")
    outFile.write("\n@DATA\n")
    for x,label,weight in zip(xs,labels,weights):
        for xj in x:
            outFile.write(str(xj) + ",")
        if label == 's':
            outFile.write(str(weight) + "," + str(-weight) + "\n")
        else:
            outFile.write(str(-weight) + "," + str(weight) + "\n")
    outFile.close()

all = list(csv.reader(open("raw/training.csv","rb"), delimiter=','))

header = np.array(all[0][1:-2])

testText = list(csv.reader(open("raw/test.csv","rb"), delimiter=','))
testIds = np.array([int(row[0]) for row in testText[1:]])
xsTest = np.array([map(float, row[1:]) for row in testText[1:]])
weightsTest = np.repeat(1.0,len(testText)-1)
labelsTest = np.repeat('s',len(testText)-1)
DataToArff(xsTest,labelsTest,weightsTest,header,"HiggsML_challenge_test","test")