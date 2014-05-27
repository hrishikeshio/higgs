import csv
import math
import copy
import gzip
from operator import itemgetter
import numpy as np
import pandas as pd

from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn import cross_validation
from sklearn.preprocessing import *
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from HiggsBosonCompetition_AMSMetric_rev1 import AMS_metric

X_train=[]
y_train=[]
X_test=[]
y_test=[]

test_mode=0
feature_selection=0
cross_validating=1
X_test_eeventids=[]

##############################Import data############################################

with open("raw/training.csv","rb") as f:
	reader=csv.reader(f)
	titletrain= reader.next()
	titleidx=range(len(titletrain))
	# print titleidx
	print titletrain
	titletrainnew= titletrain[:24]+[titletrain[27]]+titletrain[30:]
	print titletrainnew
	# print title[1:14] #consider derived features only
	for row in reader:
		# X_train.append(row[1:14]) #consider derived features only
		X_train.append(row[:24]+[row[27]]+row[30:])
		# X_train.append(row[1:24]+[row[27]]+row[30:-2])
		y_train.append(row[-1:][0])

with open("raw/test.csv","rb") as f:
	reader=csv.reader(f)
	titletest= reader.next()
	titleidx=range(len(titletest))
	# print titleidx
	titletestnew = titletest[:24]+[titletest[27]]+titletest[30:]
	print titletestnew
	for row in reader:
		# X_test.append(row[1:14])
		X_test.append(row[:24]+[row[27]]+row[30:])
		# X_test.append(row[1:24]+[row[27]]+row[30:])
		X_test_eeventids.append(row[0])
print "Input data read"


#################################Feature selection#######################################
# if feature_selection:
# 	clf = ExtraTreesClassifier()
# 	X_new = clf.fit(X_train, y_train).transform(X_train)
# 	print clf.feature_importances_  
# 	importances=dict(zip(titletrain,clf.feature_importances_ ))
# 	print importances
# 	print len(importances)
# 	print sorted(importances.items(), key=lambda x:x[1])

# 	exit()
# 	# selector = SelectPercentile(f_classif, percentile=10)
# 	# scores = -np.log10(selector.pvalues_) 
# 	# scores /= scores.max()
# 	# pl.bar(X_indices - .45, scores, width=.2,
# 	#        label=r'Univariate score ($-Log(p_{value})$)', color='g')


# print "feature selection done"
################################CLassifier Name #############################################
clf2=RandomForestClassifier()
# clf2=SVC()

# clf2 = Pipeline([
#   ('feature_selection', ExtraTreesClassifier()),
#   ('classification', RandomForestClassifier())
# ])


if cross_validating:
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
		X_train, y_train, test_size=0.4, random_state=0)
	event_ids=iter([a[0] for a in(X_test)])
	print X_train[:10]
	print X_test[:10]
	solution=X_test[:,[0,-1]]
	X_train=X_train[:,1:-2]
	X_test=X_test[:,1:-2]

if test_mode:
  clf2.fit(X_train[:10],y_train[:10])
  preds=clf2.predict(X_test[:10])
  probs=clf2.predict_proba(X_test[:10])
else:
	# clf2=LogisticRegression(C=1).fit(X_train,y_train)
	#clf2=RandomForestClassifier().fit(X_train,y_train)
	# clf2.fit(selector.transform(X_train),y_train)
	clf2.fit((X_train),y_train)
	print "Model trained"
	preds=clf2.predict(X_test)
	# preds=clf2.predict(selector.transform(X_test))
	print "prediction done"
	# probs=clf2.predict_proba(selector.transform(X_test))
	probs=clf2.predict_proba((X_test))
	print "probabilties done"
aprobs=[a[0] for a in probs]
if test_mode:
	print X_train[1]
	print X_test[1]
	print y_train[1]
	print len(X_train[1])
	print len(X_test[1])
	print y_train[1]
	print probs
	print aprobs	

ans=[]
probiter=iter(aprobs)

for a in preds:
	ans.append([event_ids.next()]+[a]+[probiter.next()])

ans_sorted=sorted(ans, key=itemgetter(2))
if test_mode:
	print "ans",ans 
	print "final", ans_sorted

length=iter(range(100000000))
ranked=[]
length.next()
for a in ans_sorted:
	ranked.append(a+[length.next()])
# print "ranked",ranked
name="submission_ours"

with open("submissions/"+name+".csv","wb") as f:
	writer=csv.writer(f)
	writer.writerow(["EventId","RankOrder","Class"])
	for row in ranked:
		writer.writerow([row[0],row[-1],row[1]])

f_in = open("submissions/"+name+".csv", 'rb')
f_out = gzip.open("submissions/"+name+".csv.gz", 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()

if cross_validating:
	with open("solution.csv", "wb") as f:
		writer=csv.writer(f)
		solutionlist=solution.tolist()
		ranks=range(len(solutionlist))
		finalsolution=[[a[0],b,a[1]] for a,b in zip(solutionlist,[a[-1] for a in ranked])]
		writer.writerow("EventId,RankOrder,Class".split(","))
		writer.writerows(finalsolution)
	print AMS_metric("solution.csv","submissions/"+name+".csv",len(solution))