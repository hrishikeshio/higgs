
# Zip the prediction so it is easier to upload.
higgs_pred.zip : higgs.pred.csv
	zip higgs_pred.zip higgs.pred.csv

# Use the model to make a prediction
higgs.pred.csv : higgs.model
	time python -u higgs-pred.py

# Use training data to make a model
higgs.model : higgs-numpy.py ../raw/training.csv
	time python -u higgs-numpy.py

clean :
	-rm -f higgs.model
	-rm -f higgs_pred.zip
	-rm -f higgs.pred.csv

