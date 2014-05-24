
import random,string,math,csv,pandas
import numpy as np
import matplotlib.pyplot as plt

resultsText = list(csv.reader(open("results.dta","rb"), delimiter='\t'))
ts = [int(result[0]) for result in resultsText]
trainErrors = np.array([float(result[5]) for result in resultsText])
testErrors = np.array([float(result[11]) for result in resultsText])

fig = plt.figure()
fig.suptitle('MultiBoost learning curves', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.set_xlabel('number of boosting iterations')
ax.set_ylabel('balanced weighted error rate')

ax.annotate('training error', xy=(0.9*len(ts), trainErrors[len(ts)-1]), 
            xytext=(0.6*len(ts), trainErrors[len(ts)-1]-0.05),
            arrowprops=dict(facecolor='blue', shrink=0.05))
ax.annotate('validation error', xy=(0.9*len(ts), testErrors[len(ts)-1]), 
            xytext=(0.6*len(ts), testErrors[len(ts)-1]+0.05),
            arrowprops=dict(facecolor='red', shrink=0.05))

ax.plot(ts,trainErrors,'b-')
ax.plot(ts,testErrors,'r-')

ax.axis([0, len(ts), 0.1, 0.3])

plt.show()

def AMS(s,b):
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return math.sqrt(2 * ((s + b + bReg) * 
                          math.log(1 + s / (b + bReg)) - s))

validationScoresText = list(csv.reader(open("scoresValidation.txt","rb"), delimiter=','))
validationScores = np.array([float(score[0]) for score in validationScoresText])
tIIs = validationScores.argsort()
wFactor = 1.* numPoints / numPointsValidation
s = np.sum(weightsValidation[sSelectorValidation])
b = np.sum(weightsValidation[bSelectorValidation])
amss = np.empty([len(tIIs)])
amsMax = 0
threshold = 0.0
for tI in range(len(tIIs)):
    # don't forget to renormalize the weights to the same sum 
    # as in the complete training set
    amss[tI] = AMS(max(0,s * wFactor),max(0,b * wFactor))
    # careful with small regions, they fluctuate a lot
    if tI < 0.9 * len(tIIs) and amss[tI] > amsMax:
        amsMax = amss[tI]
        threshold = validationScores[tIIs[tI]]
        #print tI,threshold
    if sSelectorValidation[tIIs[tI]]:
        s -= weightsValidation[tIIs[tI]]
    else:
        b -= weightsValidation[tIIs[tI]]
fig = plt.figure()
fig.suptitle('MultiBoost AMS curves', fontsize=14, fontweight='bold')
vsRank = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

vsRank.set_xlabel('rank')
vsRank.set_ylabel('AMS')

vsRank.plot(amss,'b-')

vsRank.axis([0,len(amss), 0, 4])

plt.show()
fig = plt.figure()
fig.suptitle('MultiBoost AMS curves', fontsize=14, fontweight='bold')
vsScore = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

vsScore.set_xlabel('score')
vsScore.set_ylabel('AMS')

vsScore.plot(validationScores[tIIs],amss,'b-')

vsScore.axis([validationScores[tIIs[0]],validationScores[tIIs[-1]] , 0, 4])

plt.show()
testText = list(csv.reader(open("test.csv","rb"), delimiter=','))
testIds = np.array([int(row[0]) for row in testText[1:]])
xsTest = np.array([map(float, row[1:]) for row in testText[1:]])
weightsTest = np.repeat(1.0,len(testText)-1)
labelsTest = np.repeat('s',len(testText)-1)
DataToArff(xsTest,labelsTest,weightsTest,header,"HiggsML_challenge_test","test")