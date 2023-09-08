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
  
		# STEP 1: remove HTML formatting
		cleaned_text = BeautifulSoup(text,'html.parser').get_text()

		# STEPS 2 and 3: remove non-alphabetic characters as well as leading or trailing white spaces
		cleaned_text = re.sub('^\s+|\W+|[0-9]|\s+$',' ',cleaned_text).strip()
  
		# STEP 4: convert to lower case
		cleaned_text = cleaned_text.lower()
  
		# STEP 5: tokenize and remove stopwords using nltk's 'english' vocabulary
		tokens = nltk.word_tokenize(cleaned_text)
		stopwords = nltk.corpus.stopwords.words("english")
		cleaned_tokens = [token for token in tokens if token not in stopwords]

		# STEP 6: rejoin remaining text into one string using " " as the word separator
		cleaned_text = " ".join(cleaned_tokens) if len(cleaned_tokens) > 0 else "".join(cleaned_tokens)
		return cleaned_text

	def clean_dataset(self, data):
		'''
		Given an array of strings, clean each string in the array by calling clean_text()
			
		Args:
			data: list of N strings
		
		Return:
			cleaned_data: list of cleaned N strings
		'''
  
		return [self.clean_text(s) for s in data]


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
 
	pp = Preprocess()
	return pp.clean_dataset(x_train), pp.clean_dataset(x_test)