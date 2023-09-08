import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
class Preprocess(object):
	def __init__(self):
		pass

	def clean_text(self, text):
		'''
		Clean the input text string:
			1. Remove HTML formatting
			2. Remove non-alphabet characters such as punctuation or numbers and replace with ' '
			   You may refer back to the slides for this part (We implement this for you)
			3. Remove leading or trailing white spaces including any newline characters
			4. Convert to lower case
			5. Tokenize and remove stopwords using nltk's 'english' vocabulary
			6. Rejoin remaining text into one string using " " as the word separator
			
		Args:
			text: string 
		
		Return:
			cleaned_text: string
		'''

		# Step 2 is implemented for you
		cleaned_text = re.sub('^\s+|\W+|[0-9]|\s+$',' ',cleaned_text).strip()

		raise NotImplementedError

	def clean_dataset(self, data):
		'''
		Given an array of strings, clean each string in the array by calling clean_text()
			
		Args:
			data: list of N strings
		
		Return:
			cleaned_data: list of cleaned N strings
		'''
		
		raise NotImplementedError


def clean_wos(x_train, x_test):
	'''
	ToDo: Clean both the x_train and x_test dataset using clean_dataset from Preprocess
	Input:
		x_train: list of N strings
		x_test: list of M strings
		
	Output:
		cleaned_text_wos: list of cleaned N strings
		cleaned_text_wos_test: list of cleaned M strings
	'''

	raise NotImplementedError
