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

from memory_profiler import profile

X_train = []
y_train = []
X_test = []
y_test = []
X_test_eeventids = []

# Set flags for this run 
test_mode = 0
feature_selection = 0
cross_validating = 1
 
##############################Import data#################################
# 1	EventId
# 2	DER_mass_MMC
# 3	DER_mass_transverse_met_lep
# 4	DER_mass_vis
# 5	DER_pt_h
# 6	DER_deltaeta_jet_jet
# 7	DER_mass_jet_jet
# 8	DER_prodeta_jet_jet
# 9	DER_deltar_tau_lep
# 10	DER_pt_tot
# 11	DER_sum_pt
# 12	DER_pt_ratio_lep_tau
# 13	DER_met_phi_centrality
# 14	DER_lep_eta_centrality
# 15	PRI_tau_pt
# 16	PRI_tau_eta
# 17	PRI_tau_phi
# 18	PRI_lep_pt
# 19	PRI_lep_eta
# 20	PRI_lep_phi
# 21	PRI_met
# 22	PRI_met_phi
# 23	PRI_met_sumet
# 24	PRI_jet_num
# 25	PRI_jet_leading_pt
# 26	PRI_jet_leading_eta
# 27	PRI_jet_leading_phi
# 28	PRI_jet_subleading_pt
# 29	PRI_jet_subleading_eta
# 30	PRI_jet_subleading_phi
# 31	PRI_jet_all_pt
# 32	Weight
# 33	Label

#Read data
X_train_orig = pd.read_csv("raw/training.csv")
y_train_orig = X_train_orig["Label"]
X_test_orig = pd.read_csv("raw/test.csv")
print "Input data read"

exclude=["PRI_jet_subleading_phi","PRI_jet_subleading_eta","PRI_jet_leading_phi","PRI_jet_leading_eta"]
# exclude=[]
X_train_orig = X_train_orig[[column for column in X_train_orig.columns if column not in exclude ]]
X_test_orig = X_test_orig[[column for column in X_test_orig.columns if column not in exclude ]]



# print "solution",solution[:10]
event_ids = iter(X_test_orig["EventId"])
X_train = X_train_orig.iloc[:, 1:-2]
X_test = X_test_orig.iloc[:, 1:]
y_train = y_train_orig
#######################Train model ###########################################
if test_mode:
    clf2.fit(X_train[:10], y_train[:10])
    preds = clf2.predict(X_test[:10])
    probs = clf2.predict_proba(X_test[:10])
else:
    # clf2=LogisticRegression(C=1).fit(X_train,y_train)
    # clf2=RandomForestClassifier().fit(X_train,y_train)
    # clf2.fit(selector.transform(X_train),y_train)
    clf2=RandomForestClassifier()
    clf2.fit((X_train), y_train)
    print "Model trained"
    preds = clf2.predict(X_test)
    # preds=clf2.predict(selector.transform(X_test))
    print "prediction done"
    # probs=clf2.predict_proba(selector.transform(X_test))
    probs = clf2.predict_proba((X_test)) # Get probabilities so we can predict RankOrder
    print "probabilties done"
aprobs = [a[0] for a in probs]
if test_mode:
    print X_train[1]
    print X_test[1]
    print y_train[1]
    print len(X_train[1])
    print len(X_test[1])
    print y_train[1]
    print probs
    print aprobs

ans = []
probiter = iter(aprobs)
# print len(aprobs),len(event_ids)
for a in preds:
    ans.append([event_ids.next()] + [a] + [probiter.next()])

ans_sorted = sorted(ans, key=itemgetter(2),  reverse=True)
if test_mode:
    print "ans", ans
    print "final", ans_sorted

# length = iter(range(100000000))
ranked = []
# length.next()
for a,b in zip(ans_sorted,range(1,len(ans_sorted)+1)):
    ranked.append(a + [b])
# print ranked[:10]
# print zip(ranked)
# exit()
# print "ranked",ranked
name = "submission_ours" # Output file name

with open("submissions/" + name + ".csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerow(["EventId", "RankOrder", "Class"])
    for row in ranked:
        writer.writerow([row[0], row[-1], row[1]])

f_in = open("submissions/" + name + ".csv", 'rb')
f_out = gzip.open("submissions/" + name + ".csv.gz", 'wb') # Gzip file so we can upload it quickly
f_out.writelines(f_in)
f_out.close()
f_in.close()

