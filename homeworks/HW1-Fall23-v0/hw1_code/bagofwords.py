import numpy as np
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ProcessPoolExecutor  # You may not need to use this
from functools import partial # You may not need to use this

class OHE_BOW(object): 
	def __init__(self):
		'''
		Initialize instance of OneHotEncoder in self.oh for use in fit and transform
		'''
		self.vocab_size = None 			#keep

		raise NotImplementedError

	def split_text(self, data):
		'''
		Helper function to separate each string into a list of individual words
		Args:
			data: list of N strings
		
		Return:
			data_split: list of N lists of individual words from each string
		'''
		raise NotImplementedError

	def flatten_list(self, data):
		'''
		Helper function to flatten a list of list of words into a single list
		Args:
			data: list of N lists of W_i words 
		
		Return:
			data_split: (W,) numpy array of words, 
				where W is the sum of the number of W_i words in each of the list of words		
		'''
		raise NotImplementedError

	def fit(self, data):
		'''
		Fit the initialized instance of OneHotEncoder to the given data
		Use split_text to separate the given strings into a list of words and 
		flatten_list to flatten the list of words in a sentence into a single list of words
		
		Set self.vocab_size to the number of unique words in the given data corpus
		Args:
			data: list of N strings 
		
		Return:
			None
		Hint: You may find numpy's reshape function helpful when fitting the encoder
		'''
		raise NotImplementedError


	def onehot(self, words):
		'''
		Helper function to encode a list of words into one hot encoding format
		Args:
			words: list of W_i words from a string
		
		Return:
			onehotencoded: (W_i, D) numpy array where:
				W_i is the number of words in the current input list i
				D is the vocab size
		Hint: 	.toarray() may be helpful in converting a sparse matrix into a numpy array
				You can use sklearn's built-in OneHotEncoder transform function
		'''
		raise NotImplementedError

	def transform(self, data):
		'''
		Use the already fitted instance of OneHotEncoder to help you transform the given 
		data into a bag of words representation. You will need to separate each string 
		into a list of words and iterate through each list to transform into a one hot 
		encoding format.
		Use your one hot encoding of each word in a sentence to get the bag of words count
		representation. You may want to look 
		For any empty strings append a (1, D) array of zeros instead.
			
		Args:
			data: list of N strings
		
		Return:
			bow: (N, D) numpy array
		Hint: Using a try and except block during one hot encoding transform may be helpful
		'''
		raise NotImplementedError