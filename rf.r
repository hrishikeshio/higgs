library("randomForest", lib.loc="/home/hrishikesh/R/x86_64-pc-linux-gnu-library/2.14")
train<-read.csv("training.csv")
weights<-train$Weight
train$Weight<- 0