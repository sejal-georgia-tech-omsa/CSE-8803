import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB


class MNB(object):
	def __init__(self):
		'''
		Initialize instance of MultinomialNB in self.clf for use in fit and transform
		'''
		raise NotImplementedError

	def fit(self, data, y):
	    '''
		Use the initialized instance of MultinomialNB to fit to the given data
	    Args:
	        data: (N, D) numpy array of numerically encoded sentences
	        y: (N,) numpy vector of labels 
	    
	    Return:
	        None
	    '''
	    raise NotImplementedError

	def predict(self, data):
	    '''
		Use the initialized instance of MultinomialNB to predict the label of the given data
	        
	    Args:
	        data: (N, D) numpy array of numerically encoded sentences
	    
	    Return:
	        y_hat: (N,) numpy vector of labels
	    '''
	    raise NotImplementedError


class GNB(object):
	def __init__(self):
		'''
		Initialize instance of GaussianNB in self.clf for use in fit and transform
		'''
		raise NotImplementedError

	def fit(self, data, y):
	    '''
		Use the initialized instance of GaussianNB to fit to the given data
	    Args:
	        data: (N, D) numpy array of numerically encoded sentences
	        y: (N,) numpy vector of labels 
	    
	    Return:
	        None
	    '''
	    raise NotImplementedError

	def predict(self, data):
	    '''
		Use the initialized instance of GaussianNB to predict the label of the given data
	        
	    Args:
	        data: (N, D) numpy array of numerically encoded sentences
	    
	    Return:
	        y_hat: (N,) numpy vector of labels
	    '''
	    raise NotImplementedError
