import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score

class Metrics(object):
	def __init__(self):
		pass

	def accuracy(self, y, y_hat):
		'''
		Without using sklearn, implement the accuracy metric as the ratio between the number of
		correctly predicted datapoints against total number of datapoints
		Args:
			y_hat: (N,) numpy vector of predicted labels
			y: list of N labels
		
		Return:
			accuracy: scalar
		'''
		raise NotImplementedError

	def recall(self, y, y_hat, average='macro'):
		'''
		Use sklearn's recall_score function to calculate the recall score
			
		Args:
			y_hat: (N,) numpy vector of predicted labels
			y: list of N labels
			average: None or string 'macro'
		
		Return:
			recall: scalar or list of scalars with length of # of unique labels
		'''
		raise NotImplementedError

	def precision(self, y, y_hat, average='macro'):
		'''
		Use sklearn's precision_score function to calculate the precision score
			
		Args:
			y_hat: (N,) numpy vector of predicted labels
			y: list of N labels
			average: None or string 'macro'
		
		Return:
			precision: scalar or list of scalars with length of # of unique labels
		'''
		raise NotImplementedError

	def f1_score(self, y, y_hat, average='macro'):
		'''
		Use sklearn's f1_score function to calculate the f1 score
			
		Args:
			y_hat: (N,) numpy vector of predicted labels
			y: list of N labels
			average: None or string 'macro'
		
		Return:
			f1_score: scalar or list of scalars with length of # of unique labels
		'''
		raise NotImplementedError

	def roc_auc_score(self, y, y_hat, average='macro'):
		'''
		Use sklearn's roc_auc_score function to calculate the ROC_AUC score
			
		Args:
			y_hat: (N,) numpy vector of predicted labels
			y: list of N labels
			average: None or string 'macro'
		
		Return:
			roc_auc_score: scalar or list of scalars with length of # of unique labels
		'''
		raise NotImplementedError
    
	def confusion_matrix(self, y, y_hat):
		'''
		Use sklearn's confusion_matrix function to calculate the Confusion Matrix
			
		Args:
			y_hat: (N,) numpy vector of predicted labels
			y: list of N labels
		
		Return:
			confusion_matrix: numpy array of the predictions vs ground truth counts.
		'''
		raise NotImplementedError
