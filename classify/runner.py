#!/usr/bin/env python
# coding=utf-8

"""runner.py: Class that reads the .csv config file, loads the embeddings and runs the classification system."""

__author__ 	= "Ruggero Petrolito"
__email__ 	= "ruggero.petrolito@gmail.com"


import argparse
import codecs
import os

import fasttext
import word2vec

from kfold import run


class Runner:
	"""
		When initialized, the class reads the .csv config file and stores the parameters in a dictionary.
		It loads each model specified in the config file, and stores each model as an attribute.
	"""
	tweets_w2v = None
	tweets_ft = None
	paisa_w2v = None
	paisa_ft = None
	csv_filename = None

	def __init__(self, csv_filename, results_folder):
		"""
			:param csv_filename: name of config file
			:param results_folder: folder where the results file is going to be stored
		"""
		self.csv_filename = csv_filename
		self.results_folder = results_folder

	def load_tweets_ft(self, filename):
		"""
			:param filename: name of model file
			:return: nothing
			Loads model trained on Tweets corpus using FastText. Stores it in self.tweets_ft
		"""
		print 'Loading embeddings from', filename
		self.tweets_ft = fasttext.load_model(filename)

	def load_tweets_w2v(self, filename):
		"""
			:param filename: name of model file
			:return: nothing
			Loads model trained on Tweets corpus using Word2Vec. Stores it in self.tweets_w2v
		"""
		print 'Loading embeddings from', filename
		self.tweets_w2v = word2vec.load(filename)

	def load_paisa_ft(self, filename):
		"""
			:param filename: name of model file
			:return: nothing
			Loads model trained on Paisà corpus using FastText. Stores it in self.paisa_ft
		"""
		print 'Loading embeddings from /home/ruggero/MEGA/tesi_magistrale/modelli_distribuzionali/paisa_ft.bin'
		self.paisa_ft = fasttext.load_model(filename)

	def load_paisa_w2v(self, filename):
		"""
			:param filename: name of model file
			:return: nothing
			Loads model trained on Paisà corpus using Word2Vec. Stores it in self.paisa_w2v
		"""
		print 'Loading embeddings from /home/ruggero/MEGA/tesi_magistrale/modelli_distribuzionali/paisa_w2v.bin'
		self.paisa_w2v = word2vec.load(filename)

	def run_experiments(self):
		"""
			Reads the config file and runs the classifier.
		"""
		with codecs.open(self.csv_filename, 'r', 'utf8') as infile:
			results_folder_path = self.results_folder
			print 'CHOSEN FOLDER FOR RESULTS:', '-->', self.results_folder
			try:
				"""
					Creates the folder to contain the results.
					If the folder already exists, it gets an error, so does nothing.
				"""
				os.mkdir(results_folder_path)
			except OSError:
				pass
			name = self.csv_filename.split('/')[-1]
			result_filename = results_folder_path + '/' + name[0:-4] + '.results.csv' # creates the results filename from the config filename
			results_file = codecs.open(result_filename, 'w', 'utf8')
			for line in infile:
				"""
					Each line corresponds to an experiment.
					For each line, the script reads the parameters and run the nth experiment.
				"""
				if not line[0].isdigit():
					"""
						Case of the header line of the csv file.
						Copies the header line to the result file, to keep track of used parameters.
						Adds the accuracies fields to the header line of the result file.
						Continues, because there's no need to proceed with the following steps.
					"""
					results_file.write(line.rstrip())
					for task in ('SUBJ', 'OPOS', 'ONEG', 'IRO', 'POL'):
						results_file.write(',' + task + '_FSCORE')
					continue
				"""
					Splits the line to get the parameters.
					The options to be used are marked with 'y' in the config file.
					The options not to be used are marked with 'n' in the config file.
					The 'y' and 'n' values are converted to booleans.
				"""
				fields = line.rstrip().split(',')
				for i in range(len(fields)):
					fields[i] = fields[i].strip('"')
					if fields[i].isdigit():
						fields[i] = int(fields[i])
					elif fields[i] == 'y':
						fields[i] = True
					elif fields[i] == 'n':
						fields[i] = False
				"""
					Stores the parametes in a dictionary of properties.
				"""
				properties = {
					'CLASSIFIER': 'SVM',
					'FORMAT': 'CONLL',
					'K': fields[1],
					'TWEETS_W2V': fields[2],
					'PAISA_W2V': fields[3],
					'TWEETS_FT': fields[4],
					'PAISA_FT': fields[5],
					'USE_WORD_VECTORS': fields[6],
					'SUM_VECTOR': fields[7],
					'AVERAGE_VECTOR': fields[8],
					'PRODUCT_VECTOR': fields[9],
					'MIN_POOL_VECTOR': fields[10],
					'MAX_POOL_VECTOR': fields[11],
					'POS_ALL': fields[12],
					'POS_LIST': [],
					'ALL_CAPS': fields[17],
					'CONTAINS_CAPS': fields[18],
					'POSITIVE_EMOTICONS': fields[19],
					'NEGATIVE_EMOTICONS': fields[20],
					'ELONGATED_WORDS': fields[21],
					'NEGATIONS': fields[22],
					'BAG_OF_WORDS': fields[23],
					'KERNEL': fields[24],
					'EMBEDDING_MODEL': fields[25],
					'SENTENCE_VEC_TW': fields[26],
					'SENTENCE_VEC_PA': fields[27],
					'VECTORS_SIZE': fields[28]
				}
				"""
					Populates the list of the parts of speech to be used.
				"""
				if fields[12]:
					properties['POS_LIST'].append('all')
				if fields[13]:
					properties['POS_LIST'].append('S')
				if fields[14]:
					properties['POS_LIST'].append('V')
				if fields[15]:
					properties['POS_LIST'].append('A')
				if fields[16]:
					properties['POS_LIST'].append('B')
				vector_models = []
				"""
					Each selected vector model is loaded, store in the corresponding attribute and added to the list of vector models to be used.
				"""
				if properties['TWEETS_W2V']:
					if properties['EMBEDDING_MODEL'] == 'default':
						emb_folder = '../embedding_models/'
						filename = emb_folder + 'tweets_w2v.bin'
						self.load_tweets_w2v(filename)
					else:
						filename = properties['EMBEDDING_MODEL']
						self.load_tweets_w2v(filename)
					vector_models.append(self.tweets_w2v)
				if properties['TWEETS_FT'] or properties['SENTENCE_VEC_TW']:
					if properties['EMBEDDING_MODEL'] == 'default':
						filename = emb_folder + 'tweets_ft.bin'
						self.load_tweets_ft(filename)
					else:
						filename = properties['EMBEDDING_MODEL']
						self.load_tweets_ft(filename)
					vector_models.append(self.tweets_ft)
				if properties['PAISA_W2V']:
					self.load_paisa_w2v(emb_folder + 'paisa_w2v.bin')
					vector_models.append(self.paisa_w2v)
				if properties['PAISA_FT'] or properties['SENTENCE_VEC_PA']:
					self.load_paisa_ft(emb_folder + 'paisa_ft.bin')
					vector_models.append(self.paisa_ft)
				run(properties, vector_models, results_file, line)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run classification.')
	parser.add_argument('--prop', type=str, help='Properties csv file for classification experiments.')
	parser.add_argument('--folder', type=str, help='Folder for results of the experiments.')
	args = parser.parse_args()
	runner = Runner(args.prop, args.folder)
	runner.run_experiments()
