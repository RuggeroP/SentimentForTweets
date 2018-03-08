#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	kfold.py: script for managing the cross validation.
	The main method is 'run'
"""

__author__ 	= "Ruggero Petrolito"
__email__ 	= "ruggero.petrolito@gmail.com"


import os
import re
from evaluation import evaluate

from feature_extraction import FeatureExtractor
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


def get_prediction_matrix(training_samples, training_labels_vectors, test_samples, test_id_s, test_top_s, kernel):
	"""
		creates the prediction matrix in the format required by the evaluation script

		:param training_samples:
		:param training_labels_vectors:
		:param test_samples:
		:param test_id_s: id field
		:param test_top_s: topic field
		:param kernel: kernel for SVM
		:return: prediction matrix (it's a list of vector, a vector for each tweet)
	"""

	vectors = [test_id_s]
	print 'Using kernel', kernel
	# classifier = SVC(C=1, kernel=kernel, max_iter=1000)
	if kernel == 'linear':
		classifier = LinearSVC(C=1)
	else:
		classifier = SVC(C=1, kernel=kernel)
	for labels_vector in training_labels_vectors:
		classifier.fit(training_samples, labels_vector)
		vectors.append(classifier.predict(test_samples))
	vectors.append(test_top_s)
	return zip(*vectors)


def get_gold_matrix(test_labels_vectors, test_id_s, test_top_s):
	"""
		creates the gold matrix as required by the evaluation script

		:param test_labels_vectors:
		:param test_id_s:
		:param test_top_s:
		:return: gold matrix (list of vectors, one for each tweet)
	"""
	vectors = [test_id_s]
	for v in test_labels_vectors:
		vectors.append(v)
	vectors.append(test_top_s)
	return zip(*vectors)


def matrix2string(matrix):
	"""
		converts matrix from list to string, as required by evaluation script
		:param matrix: prediction or gold matrix
		:return: matrix (as string)
	"""
	lines = ''
	for entry in matrix:
		s = ''
		for j, field in enumerate(entry):
			if j > 0:
				s += ','
			s += '"' + str(field) + '"'
		lines += s + '\n'
	return lines


def write_results(results_file, results_dict):
	results_file.write(',')
	results_file.write(str(results_dict['subj']))
	results_file.write(',')
	results_file.write(str(results_dict['opos']))
	results_file.write(',')
	results_file.write(str(results_dict['oneg']))
	results_file.write(',')
	results_file.write(str(results_dict['iro']))
	results_file.write(',')
	results_file.write(str(results_dict['polarity']))
	'''
	for key, value in results_dict.iteritems():
		if key == 'polarity':
			results_file.write(',' + str(results_dict['polarity']))
		else:
			for new_value in results_dict[key].values():
				results_file.write(',' + str(new_value))
	'''


def fill_dicts(samples, dicts, next_word_position):
	"""
		for each tweet, it creates a vector with the number of occurrences in the tweet of each unique word seen in the training set;
		extends the corresponding sample with the vector

		:param samples: training or test samples
		:param dicts: dictionaries of word occurrences
		:param next_word_position: number of unique words seen in training-set
		:return: nothing
	"""
	# fills the dictionaries with 0 in the empty positions
	for i in range(len(samples)):
		extension = [0.0] * next_word_position
		for j in range(next_word_position):
			try:
				extension[j] = dicts[i][j]
			except KeyError:
				pass
		samples[i].extend(extension)


def run(properties, vector_models, results_file, csv_line):
	"""
		Script that runs the k-fold

		:param properties: dictionary containing the parameters specified in the config file for the current experiment
		:param vector_models: list of embedding models to be used in this experiment
		:param results_file: csv file where the accuracies are going to be written
		:param csv_line: line of csv config file corresponding to the current experiment
		:return: nothing
	"""
	kfold_folder_path = '../data/kfold/' # folder containing the k partitions (the development set has already been split during preprocessing)
	print 'Writing to output file:', results_file.name
	k = properties['K']
	results_file.write(csv_line.rstrip()) # it copies the line from config file to keep track of used parameters
	r = range(1, k + 1)
	"""
		If the bag-of-words feature is chosen, the script initializes dictionary for the list of all words in the data-set.
		It also initializes the index value available for the next unseen word, with the value of zero because no word has been added yet.
		This index value will be updated every time an unseen word occurs in the data-set.
	"""
	if properties['BAG_OF_WORDS']:
		feature_extractor = FeatureExtractor(properties, words_dict={}, next_word_position=0, vector_models=vector_models, vectors_size=properties['VECTORS_SIZE'])
	else:
		feature_extractor = FeatureExtractor(properties, vector_models=vector_models, vectors_size=properties['VECTORS_SIZE'])
	results_dict = {
		'subj': 0.0,
		'opos': 0.0,
		'oneg': 0.0,
		'iro': 0.0,
		'polarity': 0.0
	}
	kfold_folder_path += 'conll/'
	"""
		managing cross validation;
		the k partition have already been created during pre-processing
	"""
	if k > 1:
		for i in r:
			print
			print 'RUNNING ITERATION N.', str(i)
			kth_value_folder = kfold_folder_path + str(k) + '/'
			""" creates list of partition sorted by k value inside file names """
			partitions = sorted(os.listdir(kth_value_folder), key=lambda x: (int(re.sub('\D', '', x)), x))
			test_file = kth_value_folder + 'fold_' + str(i)
			for index in range(len(partitions)):
				partitions[index] = kth_value_folder + partitions[index]
			partitions.pop(partitions.index(test_file))
			try:
				assert len(partitions) == k - 1
			except AssertionError:
				print 'Error: invalid number of partitions'
			""" samples with word, emoj and embedding features """
			training_samples = []
			""" dictionaries of word occurrences in tweets for bag-of-words """
			training_words_dicts = []
			""" dictionary with training labels """
			training_labels = {
				'subj_s': [],
				'opos_s': [],
				'oneg_s': [],
				'iro_s': [],
				'lpos_s': [],
				'lneg_s': []
			}
			extraction_function = feature_extractor.extract_from_conll
			""" using partitions as training set, with the exception of kth one """
			for training_file in partitions:
				samples, dicts, subj_s, opos_s, oneg_s, iro_s, lpos_s, lneg_s = extraction_function(training_file)
				training_labels['subj_s'] += subj_s
				training_labels['opos_s'] += opos_s
				training_labels['oneg_s'] += oneg_s
				training_labels['iro_s'] += iro_s
				training_labels['lpos_s'] += lpos_s
				training_labels['lneg_s'] += lneg_s
				training_samples += samples
				training_words_dicts += dicts
			""" sets to zero the empty positions in bags-of-words of training tweets """
			fill_dicts(training_samples, training_words_dicts, feature_extractor.next_word_position)
			test_labels = {}
			""" using kth partition as test-set """
			samples, test_words_dicts, id_s, top_s, subj_s, opos_s, oneg_s, iro_s, lpos_s, lneg_s = extraction_function(test_file, test=True)
			""" delete embedding models """

			test_samples = samples
			test_labels['subj_s'] = subj_s
			test_labels['opos_s'] = opos_s
			test_labels['oneg_s'] = oneg_s
			test_labels['iro_s'] = iro_s
			test_labels['lpos_s'] = lpos_s
			test_labels['lneg_s'] = lneg_s
			test_labels['id_s'] = id_s
			""" sets to zero the empty positions in bags-of-words of test tweets """
			fill_dicts(test_samples, test_words_dicts, feature_extractor.next_word_position)
			training_labels_vectors = [
				training_labels['subj_s'],
				training_labels['opos_s'],
				training_labels['oneg_s'],
				training_labels['iro_s'],
				training_labels['lpos_s'],
				training_labels['lneg_s']
			]
			test_labels_vectors = [
				test_labels['subj_s'],
				test_labels['opos_s'],
				test_labels['oneg_s'],
				test_labels['iro_s'],
				test_labels['lpos_s'],
				test_labels['lneg_s']
			]
			test_id_s = test_labels['id_s']
			predict_matrix = get_prediction_matrix(training_samples, training_labels_vectors, test_samples, test_id_s, top_s, properties['KERNEL'])
			gold_matrix = get_gold_matrix(test_labels_vectors, test_id_s, top_s)
			prediction_lines = matrix2string(predict_matrix)
			test_lines = matrix2string(gold_matrix)

			""" write prediction and gold matrix to file for the evaluation script"""
			tmp_folder = '../tmp/'
			tmp_result_file = open(tmp_folder + 'tmp_res.txt', 'w')
			tmp_gold_file = open('tmp_folder + tmp_gold.txt', 'w')
			tmp_result_file.write(prediction_lines)
			tmp_gold_file.write(test_lines)
			tmp_result_file.close()
			tmp_gold_file.close()

			""" evaluate and write accuracies to temporary file"""
			tmp_out_file_name = 'tmp_out' + str(i) + '.txt'
			tmp_out_file = open(tmp_out_file_name, 'w')
			evaluate('tmp_res.txt', 'tmp_gold.txt', outfile=tmp_out_file, verbose=False)
			tmp_out_file.close()
			""" parse temporary results file and updates the dictionary with experiment results"""
			with open(tmp_out_file_name, 'r') as infile:
				task = ''
				for line in infile:
					if 'task' in line:
						task = line.rstrip().split()[-1]
					if line[0].isdigit():
						""" add the accuracies values to the dictionary of accuracies """
						results_dict[task] += float(line.rstrip().split()[-1])
		for key, value in results_dict.iteritems():
			""" averages the results """
			results_dict[key] = value / k
	elif k == 1:
		""" if k == 1 it uses the official test-set as test """
		training_file_name = '/home/ruggero/MEGA/tesi_magistrale/classification/data/training_all.parsed'
		test_file_name = '/home/ruggero/MEGA/tesi_magistrale/classification/data/testset_annotated.parsed'
		training_labels = {
			'subj_s': [],
			'opos_s': [],
			'oneg_s': [],
			'iro_s': [],
			'lpos_s': [],
			'lneg_s': []
		}
		extraction_function = feature_extractor.extract_from_conll
		training_samples, training_words_dicts, subj_s, opos_s, oneg_s, iro_s, lpos_s, lneg_s = extraction_function(training_file_name)
		training_labels['subj_s'] += subj_s
		training_labels['opos_s'] += opos_s
		training_labels['oneg_s'] += oneg_s
		training_labels['iro_s'] += iro_s
		training_labels['lpos_s'] += lpos_s
		training_labels['lneg_s'] += lneg_s
		fill_dicts(training_samples, training_words_dicts, feature_extractor.next_word_position)
		test_labels = {}
		test_samples, test_words_dicts, id_s, top_s, subj_s, opos_s, oneg_s, iro_s, lpos_s, lneg_s = extraction_function(test_file_name, test=True)
		test_labels['subj_s'] = subj_s
		test_labels['opos_s'] = opos_s
		test_labels['oneg_s'] = oneg_s
		test_labels['iro_s'] = iro_s
		test_labels['lpos_s'] = lpos_s
		test_labels['lneg_s'] = lneg_s
		test_labels['id_s'] = id_s
		fill_dicts(test_samples, test_words_dicts, feature_extractor.next_word_position)
		training_labels_vectors = [
			training_labels['subj_s'],
			training_labels['opos_s'],
			training_labels['oneg_s'],
			training_labels['iro_s'],
			training_labels['lpos_s'],
			training_labels['lneg_s']
		]
		test_labels_vectors = [
			test_labels['subj_s'],
			test_labels['opos_s'],
			test_labels['oneg_s'],
			test_labels['iro_s'],
			test_labels['lpos_s'],
			test_labels['lneg_s']
		]
		test_id_s = test_labels['id_s']
		predict_matrix = get_prediction_matrix(training_samples, training_labels_vectors, test_samples, test_id_s, top_s, properties['KERNEL'])
		gold_matrix = get_gold_matrix(test_labels_vectors, test_id_s, top_s)
		prediction_lines = matrix2string(predict_matrix)
		test_lines = matrix2string(gold_matrix)
		tmp_result_file = open('tmp_res.txt', 'w')
		tmp_gold_file = open('tmp_gold.txt', 'w')
		tmp_result_file.write(prediction_lines)
		tmp_gold_file.write(test_lines)
		tmp_result_file.close()
		tmp_gold_file.close()
		tmp_out_file_name = 'tmp_out' + '.txt'
		tmp_out_file = open(tmp_out_file_name, 'w')
		evaluate('tmp_res.txt', 'tmp_gold.txt', outfile=tmp_out_file, verbose=False)
		tmp_out_file.close()
		with open(tmp_out_file_name, 'r') as infile:
			task = ''
			for line in infile:
				if 'task' in line:
					task = line.rstrip().split()[-1]
				if line[0].isdigit():
					results_dict[task] += float(line.rstrip().split()[-1])
	write_results(results_file, results_dict)
