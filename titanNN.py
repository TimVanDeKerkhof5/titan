import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
import numpy as np
import tensorflow as tf
import datetime as dt

from fancyimpute import KNN
from scipy import stats

from sklearn import preprocessing as skp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#filepaths (added to a dictionary for easy access & expansion:
pathdict = {'cardiologie':'/home/kerkt02/patdata/DM_CARDIOLOGIE.csv', 'subtraject':'/home/kerkt02/patdata/QUERY_FOR_DM_SUBTRAJECTEN.csv','opname':'/home/kerkt02/patdata/QUERY_FOR_DM_LGS_OPNAME.csv'}

#dataframes: dfcardio, sub, admission and raw are not used when not initializing, they should be hashed to spare compution time.
#dfcardio = pd.read_csv(pathdict['cardiologie'],header=0,low_memory=False,encoding='ISO-8859-1')
#dfsub = pd.read_csv(pathdict['subtraject'],header=0,low_memory=False,encoding='ISO-8859-1')
#dfadmission = pd.read_csv(pathdict['opname'],header=0,low_memory=False,encoding='ISO-8859-1')
#dfread = pd.DataFrame()
nnraw = pd.read_csv('data.csv',header=0,low_memory=False,encoding='ISO-8859-1')
nninput = pd.read_csv('input.csv',header=0)
col = nninput.columns

#constants:
geb = 'Geboortedatum'
selectedcon = ['Geboortedatum', 'lengte','gewicht','bloeddruk','HB','HT','INR','Glucose','Kreat','Trombocyten','Leukocyten','Cholesterol_totaal','Cholesterol_ldl']
selectedcat = ['Geslacht','DiagnoseCode','vrgeschiedenis_myochardinfarct','vrgeschiedenis_PCI','vrgeschiedenis_CABG','vrgeschiedenis_CVA_TIA','vrgeschiedenis_vaatlijden','vrgeschiedenis_hartfalen','vrgeschiedenis_maligniteit','vrgeschiedenis_COPD','vrgeschiedenis_atriumfibrilleren','TIA','CVA_Niet_Bloedig','CVA_Bloedig','dialyse','riscf_roken','riscf_familieanamnese','riscf_hypertensie','riscf_hypercholesterolemie','riscf_diabetes','roken','Radialis','Femoralis','Brachialis','vd_1','vd_2','vd_3']
concatlist = selectedcon+selectedcat

selectedtarget = 'lbl'

#CNN parameters:
learningrateNN = 0.003
shapeNN = len(nninput.columns)-1
stepsNN = 20000


#split data to train and test .85 - .15
def initForCNN():
	dftest, dftrain = train_test_split(nninput, test_size = 0.85)
	cnntrain = dftrain[dftrain.columns.difference([selectedtarget])]
	cnntest = dftest[dftest.columns.difference([selectedtarget])]
	cnnlbltrain = dftrain[selectedtarget].values
	cnnlbltest = dftest[selectedtarget].values
	return cnntrain, cnntest, cnnlbltrain, cnnlbltest

#this method creates a dataframe with variables needed for the determination of readmission. done like this to improve editability
#P, Z, O for patient, ziektegeval and Opnameziektegeval.
def dataFrameBuilder(P, Z, O):
	cardioPatients = set(dfcardio[P])
	subZ = set(dfsub[Z])
	subO = set(dfadmission[O])
	diseaseIDs = set(subZ & subO)

	#init of empty lists that hold data for the dataframe.
	patnrlist = []
	opnamelist = []
	ontslaglist = []
	zorgtrajectnrlist = []

	for id in diseaseIDs:
		for i, r in dfadmission.loc[dfadmission[O] == id].iterrows():
			patnrlist.append(r[P])
			opnamelist.append(r['opname_dt'])
			ontslaglist.append(r['ontslag_dt'])

			uniquenr = set()
			uniquenr.clear()
			for ii, rr in dfsub.loc[dfsub[Z] == id].iterrows():
				uniquenr.add(rr['ZORGTRAJECTNR'])
			if(len(uniquenr) > 1):
				zorgtrajectnrlist.append(list(uniquenr)[1])
			else:
				zorgtrajectnrlist.append(list(uniquenr)[0])

	readdf = pd.DataFrame()
	readdf['Patnr'] = patnrlist
	readdf['opname_dt'] = opnamelist
	readdf['ontslag_dt'] = ontslaglist
	readdf['zorgtrajectnr'] = zorgtrajectnrlist
	readdf = readdf.sort_values(by=['Patnr','zorgtrajectnr'])
	return readdf

#this method takes a dataframe, determines readmission based on given dates, returns a list of patient IDs that are positive for readmission.
def checkForReadmission(df):
#initialization of variables used in a couple of logic gates in this method (it's a tad cluttered).
	patnr = 1
	zCode = 0
	prevzCode = 0

	readm = False
	patchecker = True
	z = False

	readTrue = []

	prevdischarge = dt.datetime(100, 10, 10).date()
	prevadmission = prevdischarge

	for index, row in df.iterrows():
		admission = row['opname_dt'][0:-9]
		admission = dt.datetime.strptime(admission, '%d%b%y').date()

		discharge = row['ontslag_dt'][0:-9]
		discharge = dt.datetime.strptime(discharge, '%d%b%y').date()
		zCode = row['zorgtrajectnr']

		if patnr != row['Patnr']:
			patchecker = True
			if patnr == 1:
				print('Successful initialization.')
			elif readm:
				readTrue.append(patnr)
				readm = False
			elif not readm:
				readm = False
			patnr = row['Patnr']

		if admission != prevadmission:
			patchecker = True

		if patchecker:
			if(abs(prevdischarge - admission).days) < 30 and z:
				readm = True
			prevdischarge = discharge
			prevadmission = admission
			patchecker = False
		z = False
		if zCode == prevzCode:
			z = True
		prevzCode = zCode

	if readm:
		readTrue.append(patnr)

	return readTrue

#random forest classifier
def rfc(features, target):
	cl = RandomForestClassifier()
	cl.fit(features,target)
	return cl

#gradient boosting classifier (uses algoritm similar to optimizer used in CNN
def gtc(features, target):
	clf = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.3,max_depth=1,random_state=0)
	clf.fit(features,target)
	return clf

#this method holds the configuration for a convolutional neural network. It is called in a different method.
def CNN(features, labels, mode):

#input layer
	input_layer = tf.reshape(features["x"], [-1, shapeNN, 1, 1])

# Convolutional Layer #1, 5x1 filter
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 1],
		padding="same",
		activation=tf.nn.relu)

  # Pooling Layer #1, 2x1 filter
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=2)

  # Convolutional Layer #2, 5x1 filter
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 1],
		padding="same",
		activation=tf.nn.relu)

  # Pooling Layer #2, 2x1 filter

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 1], strides=2)

  # Flatten input to vector
	pool2_flat = tf.reshape(pool2, [- 1, int(shapeNN/4) * 1 * 64])

  # Dense Layer, 2000 neurons
	dense = tf.layers.dense(inputs=pool2_flat, units=2000, activation=tf.nn.relu)

  # Logits layer
	logits = tf.layers.dense(inputs=dense, units=2)

	predictions = {
      #get predictions with eval set)
		"classes": tf.argmax(input=logits, axis=1),
      # add softmax function
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrateNN)
		train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics for accuracy
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
		labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#imputes continuous variables, using the fancy impute package. run before normalizing
def imputer():
	precondf = pd.DataFrame(nnraw[selectedcon])
	condf = pd.DataFrame(KNN(3).complete(nnraw[selectedcon]))
	condf.columns = precondf.columns
	condf.index = precondf.index
	condf = condf[(np.abs(stats.zscore(condf)) < 10).all(axis=1)]
	return condf

#this method normalizes all given continuous variables. Encoded variables are appended to a separate pandas dataframe. also removes outliers.
def normalizer(cdf):
	condf = pd.DataFrame()
	for feature in cdf:
		print(feature)
		cdf[feature] = normalizeData(cdf[feature])
		cdf[feature] = cdf[feature].astype(np.float32)
	condf = cdf
	return condf

#function for normalization per column
def normalizeData(df):
	x = df.values.astype(float)
	x = pd.DataFrame(x)
	mms = skp.MinMaxScaler()
	x_s = mms.fit_transform(x)
	df = pd.DataFrame(x_s)
	return df

#this method one_hot encodes all given categorical variables. Encoded variables are appended to a separate pandas dataframe.
def oneHotEncoder():
	catdf = pd.DataFrame()
	for feature in selectedcat:
		dummies = pd.get_dummies(nnraw[feature],prefix=feature)
		catdf[dummies.columns] = dummies.astype(np.float32)
	return catdf

#this method converts a date of birth into age.
def DOBConverter():
	now = pd.Timestamp(dt.datetime.now())
	nnraw[geb] = pd.to_datetime(nnraw[geb], format='%Y-%m-%d')
	nnraw[geb] = nnraw[geb].where(nnraw[geb] < now, nnraw[geb] - np.timedelta64(100, 'Y'))
	nnraw[geb] = (now - nnraw[geb]).astype('<m8[Y]')


#this method strips a dataframe of duplicates and adds a lbl that points at readmission. it should be called before imputation/normalization/encoding.
def readmissionDeterminator(pat):
	lbl = []
	for i in dfcardio['PATNR'].values:
		if i in pat:
			lbl.append(1)
		else:
			lbl.append(0)
	dfcardio['lbl'] = lbl

	prevpatid = 0
	patid = 0
	for index, row in dfcardio.iterrows():
		patid = row['PATNR']
		if patid != prevpatid:
			pass
		else:
			dfcardio.drop(index, inplace=True)
		prevpatid = patid
	dfcardio.to_csv(path_or_buf='data.csv',index=False)

#this method builds & exports a dataframe out of 3 components: a label, continuous and categorical headers.
def dfexport(cat, con):
	dflbl = nnraw[selectedtarget]
	expdf = pd.concat([cat,con,dflbl],axis=1,ignore_index=False)
	expdf.dropna(how='any',inplace=True)
	print(expdf)
	expdf.to_csv(path_or_buf='input.csv',index=False)
	return expdf

#this method sets up a neural network and runs calculations.
def runNN(train_data, eval_data, train_labels, eval_labels):
	cnn_classifier = tf.estimator.Estimator(
		model_fn=CNN, model_dir="/tmp/titanNN")

	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True)
	cnn_classifier.train(
		input_fn=train_input_fn,
		steps=stepsNN,
		hooks=[logging_hook])

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)

	eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

#export step for the random forest, gauging volatility of the feature importances
def treefile(t, filename):
	importances = []
	importances = t.feature_importances_
	indices = np.argsort(importances)
	with open(filename, 'a') as file:
		for index in indices:
			file.write(col[index]+" ")
		file.write('\n')


#main
def main():
	#hash the following 3 methods when data is already exported to csv to save computation time.
#	dfread = dataFrameBuilder('PATNR', 'ZIEKTEGEVALNUMMER', 'OpnameZiektegeval')
#	idlist = checkForReadmission(dfread)
#	readmissionDeterminator(idlist)

	#hash the following 5 methods to stop the imputation/normalization/onehotencoding of data, to save computation time.
	#DOBConverter()
	#df = imputer()
	#catdf = oneHotEncoder()
	#condf = normalizer(df)
	#inputdf = dfexport(catdf, condf)

	train, test, trainlbl, testlbl = initForCNN()
	trained_model = rfc(train, trainlbl)
	trainn = np.asmatrix(train,dtype=np.float32)
	trainnlbl = np.asarray(trainlbl, dtype=np.int32)
	testt = np.asmatrix(test,dtype=np.float32)
	testtlbl = np.asarray(testlbl, dtype=np.int32)

	runNN(trainn, testt, trainnlbl, testtlbl)
	gradientmodel = gtc(train, trainlbl)
	print("gradient boost:")
	print(gradientmodel.score(test, testlbl))
	predictions = trained_model.predict(test)
	print("random forest:")
	print(accuracy_score(testlbl, predictions))
	treefile(trained_model, 'rtitan.txt')
	treefile(gradientmodel, 'gtitan.txt')

main()

