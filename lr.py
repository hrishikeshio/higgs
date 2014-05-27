import csv
import math
import copy
import gzip
from operator import itemgetter
import numpy as np

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

test_mode=1
feature_selection=1
X_test_eeventids=[]

##############################Import data############################################

with open("raw/training.csv","rb") as f:
	reader=csv.reader(f)
	titletrain= reader.next()
	titleidx=range(len(titletrain))
	# print titleidx
	 
	titletrain= titletrain[1:-2]
	print titletrain
	# print title[1:14] #consider derived features only
	for row in reader:
		# X_train.append(row[1:14]) #consider derived features only
		X_train.append(row[1:-2])
		y_train.append(row[-1:][0])

with open("raw/test.csv","rb") as f:
	reader=csv.reader(f)
	titletest= reader.next()
	titleidx=range(len(titletest))
	# print titleidx
	titletest=titletest[1:]
	print titletest
	for row in reader:
		# X_test.append(row[1:14])
		X_test.append(row[1:])
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
event_ids=iter(X_test_eeventids)
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
	writer.writerow(["EventId","Class","RankOrder"])
	for row in ranked:
		writer.writerow(row[:2]+row[-1:])

f_in = open("submissions/"+name+".csv", 'rb')
f_out = gzip.open("submissions/"+name+".csv.gz", 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()