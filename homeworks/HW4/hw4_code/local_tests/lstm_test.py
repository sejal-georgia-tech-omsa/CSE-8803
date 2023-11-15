import numpy as np
from torch import Tensor

class LSTM_Test():
	def __init__(self):

		# Sample input for CNN: 
		self.input_sequences = ("19 Things Anyone Who's Best Friends With Their Mum As An Adult Will Understand\n",
								'6.2 magnitude earthquake hits northern Chile\n',
								'Which Of The Great Lakes Are You\n')
		# sample outputs: 1 = clickbait, 0 = not clickbait
		self.output_labels = (1, 0, 1)
		
		self.output = Tensor([[ 0.1136, -0.1024], [-0.0111, -0.1475], [ 0.0088, -0.2110]])

