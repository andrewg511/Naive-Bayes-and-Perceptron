# naiveBayes.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.


import util
import ClassificationMethod
import math
import time

class NaiveBayesClassifier(ClassificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        betterAc = -1 

        cPri = util.Counter() 
        cCondProb = util.Counter()
        cCnt = util.Counter() 
	
        j = 0
        while j < len(trainingData):
			
	    datum = trainingData[j]
	    label = trainingLabels[j]
	    cPri[label] += 1
			
	    for feat, value in datum.items():
				
		cCnt[(feat,label)] += 1
				 
		if value > 0: 
		    cCondProb[(feat, label)] += 1
					 
	    j = j + 1
			
			
	j = 0
	while j < len(kgrid): 
           
	    condProb = util.Counter()
            for number, value in cCondProb.items():
                condProb[number] = condProb[number] + value
		
	    h = 0
	    pri = util.Counter()
	    while h < len(cPri.items()):
		(number, value) = cPri.items()[h]
		pri[number] = pri[number] + value
		h = h + 1 
	    
	    cnt = util.Counter()
	    for number, value in cCnt.items():
		cnt[number] = cnt[number] + value
	    
	    i = 0
            while(i < len(self.legalLabels)):
		l = 0
                while(l < len(self.features)):
                    condProb[ (self.features[l], self.legalLabels[i])] = condProb[ (self.features[l], self.legalLabels[i])] + kgrid[j]
                    cnt[(self.features[l], self.legalLabels[i])] = cnt[(self.features[l], self.legalLabels[i])] + 2*kgrid[j] 
		    l = l + 1
		i = i + 1

            
            pri.normalize()
            for m, count in condProb.items():
                condProb[m] = count * 1.0 / cnt[m]

            self.pri = pri
	    self.condProb = condProb
            predictions = self.classify(validationData)
	    acc = 0
	    
	    for i in range(len(validationLabels)):
		if predictions[i] == validationLabels[i]:
		    acc += 1
            	
            percentage = str((100.0*acc)/len(validationLabels))
            
            print("Performance on validation set for k="+ str(kgrid[j]) + ": (" + percentage + "%)") 
          
            if acc > betterAc:
		
                goodPar = (pri, condProb, kgrid[j])
                betterAc = acc
		
	    j = j + 1
	    
        self.pri, self.condProb, self.k = goodPar
	
    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logArray = util.Counter()
	i = 0
        while i < len(self.legalLabels):
	    
            logArray[self.legalLabels[i]] = math.log(self.pri[self.legalLabels[i]])
            for f, val in datum.items():
                if val <= 0:
                    logArray[self.legalLabels[i]] += math.log(1-self.condProb[f,self.legalLabels[i]])
                else:
		    logArray[self.legalLabels[i]] += math.log(self.condProb[f,self.legalLabels[i]])
		    
	    i = i + 1

        return logArray

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featArray = []
	i = 0
	while i < len(self.features):
            featArray.append((self.condProb[self.features[i], label1]/self.condProb[self.features[i], label2], self.features[i]))
	    i += 1
        featArray.sort()
        featArray = [feat for val, feat in featArray[-100:]]

        return featArray
    
