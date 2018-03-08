#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	feature_extraction.py: it extracts the feature from the data-set.
	The main method is 'extract_from_conll'
"""

__author__ 	= "Ruggero Petrolito"
__email__ 	= "ruggero.petrolito@gmail.com"


import codecs
import re

import operator

import fasttext


class FeatureExtractor:
	"""
		Class that performs the feature extraction.
	"""
	properties = None
	vector_models = []
	vector_size = 0
	words_dict = None # dictionary that stores tokens as keys and indexes as values
	"""
		index available for the next occurring unseen word in words_dict
		will be used and increased each time an unseen word occurs in training-set
	"""
	next_word_position = None

	"""
		Word and emoj features
	"""
	allCapTweets = 0
	containsCaps = 0
	positiveEmo = 0
	negativeEmo = 0
	elongated = 0
	negations = 0

	"""
		regex patterns for words with caps
	"""
	apostr_pattern = re.compile("^[a-zA-Z]+'.+$")
	caps_pattern = re.compile("^[A-ZAÈÌÒÙÁÉÍÓÚ]+[àèìòù]?$")
	no_punct_pattern = re.compile('[a-z]', re.U)
	quote_pattern = re.compile('"[^"]+"')


	"""
		regex patterns for positive emoticons
	"""

	pos_emoj_pattern = re.compile('[X:;][\‑\-]?[\)D]+|<3', re.U|re.M)
	heart_pattern = re.compile('❤️|💛|💚|💙|💜|💔|❣️|💕|💞|💓|💗|💖|❤️‍|♥', re.U|re.M)
	smile_pattern = re.compile('😀|😃|😄|☺️|😊|😇|🙂|😉|😍|😘|😗|😙|😚|😋', re.U|re.M)


	"""
		regex patterns for negative emoticons
	"""


	neg_emoj_pattern = re.compile('[:][\-\‑]?\(', re.U | re.M)
	sad_pattern = re.compile('😞|😟|🙁|☹|️😣|😖|😫|😩|😤|😠|😡|😯|😦|😧|😭|😓', re.U | re.M)

	"""
		regex pattern for elongated words
	"""

	elong_word_pattern = re.compile('[a-zA-Z]*([a-zA-Z])\\1{2,}$')

	"""
		regex pattern for negations
	"""

	negations_string = 'non|nessuno|niente|nessuna|nessun|no|mancanza|assenza|nulla|nemmeno|né|nè'
	negations_pattern = re.compile(negations_string, re.IGNORECASE)

	"""
		regex patterns for urls, mentions and hashtags
	"""

	url_pattern = re.compile('www|http')
	mention_pattern = re.compile('^@.+$')
	hashtag_pattern = re.compile('^#.+$')

	def __init__(self, properties, vector_models=[], vectors_size=None, words_dict={}, next_word_position=0):
		"""
			Initializes the class instance.

			:param properties: dictionary that contains the parameters specified in the config csv
			:param vector_models: list of embedding models to use in the feature extraction
			:param vectors_size: chosen size for the embeddings
			:param words_dict: dictionary for the bag-of-words
			:param next_word_position: index for the next unseen word (when using bag-of-words)
		"""
		""" using '__class__' for the following 2 attributes, because we want to keep track of them for the test"""
		self.__class__.words_dict = words_dict
		self.__class__.next_word_position = next_word_position

		self.vector_models = vector_models
		self.vectors_size = vectors_size
		self.properties = properties
		""" initializing a list of zeros to maybe use later """
		if self.properties['USE_WORD_VECTORS']:
			self.zero_vector = [0] * self.vectors_size
		print 'Initialized FeatureExtractor'
		print 'Using', len(self.vector_models), 'models'
		print 'Vectors size:', self.vectors_size

	def analyze_token(self, token, tweet_words_dict, test):
		"""
		Analyzes the token

		:param token: input token
		:param tweet_words_dict: dictionary of occurrences of words (stored as indexes) in tweet
		:param test: boolean, true if the file we analyze is the test set, false otherwise
		:return: the features based on word-shape or emoj type and the embeddings
		"""
		token_morpho_features = {
			'caps_count': 0,
			'non_caps_count': 0,
			'pos_emoj_count': 0,
			'neg_emoj_count': 0,
			'elong_words_count': 0,
			'negations_count': 0
		}
		""" extraction of chosen word-based and emoj-based features """
		if self.properties['ALL_CAPS'] or self.properties['CONTAINS_CAPS']:
			is_caps = bool(self.caps_pattern.match(token))
			hashtag = self.hashtag_pattern.match(token)
			url = self.url_pattern.match(token)
			mention = self.mention_pattern.match(token)
			token_morpho_features['caps_count'] += is_caps
			if bool(re.search('\w', token)):
				if not re.match('^\d+$', token):
					if not re.match('^_$', token):
						if not re.match('^[àèìòùáéíóúÀÈÌÒÙÁÉÍÓÚ]$', token):
							if len(token) > 2:
								if not (hashtag or url or mention):
									if re.search('[a-z]', token):
										token_morpho_features['non_caps_count'] += 1
		if self.properties['POSITIVE_EMOTICONS']:
			token_morpho_features['pos_emoj_count'] += bool(self.pos_emoj_pattern.match(token) or self.heart_pattern.search(token) or self.smile_pattern.search(token))
		if self.properties['NEGATIVE_EMOTICONS']:
			token_morpho_features['neg_emoj_count'] += bool(self.neg_emoj_pattern.match(token) or self.sad_pattern.search(token))
		if self.properties['ELONGATED_WORDS']:
			token_morpho_features['elong_words_count'] += bool(self.elong_word_pattern.search(token))
		if self.properties['NEGATIONS']:
			token_morpho_features['negations_count'] += bool(self.negations_pattern.match(token))
		""" the length of token_vectors depends on how many models are used (tweet_w2v, tweet_ft, etc...) """
		token_vectors = []
		""" extraction of embeddings """
		if self.properties['USE_WORD_VECTORS']:
			if self.url_pattern.match(token):
				token = '_URL_'
			elif self.hashtag_pattern.match(token):
				token = '_TAG_'
			elif self.mention_pattern.match(token):
				token = '_USER_'
			for i, vector_model in enumerate(self.vector_models):
				try:
					vec = vector_model[token]
					if 'array' in str(type(vec)):
						vec = vec.tolist()
					token_vectors.append(vec)
				except KeyError:
					pass
		if self.properties['BAG_OF_WORDS']:
			if self.url_pattern.match(token):
				token = '_URL_'
			elif self.hashtag_pattern.match(token):
				token = '_TAG_'
			elif self.mention_pattern.match(token):
				token = '_USER_'
			try:
				""" checks word index in overall dictionary (key is word, value is index)"""
				word_position = self.words_dict[token]
				try:
					"""
						tweet dictionary is updated;
						value of occurrences of word in tweet is increased;
						key is word position in overall dictionary;
						value is the number of occurrences of word in tweet
					"""
					tweet_words_dict[word_position] += 1
				except KeyError:
					""" word hasn't occurred in tweet yet """
					tweet_words_dict[word_position] = 0
			except KeyError:
				""" word hasn't occurred in training-set yet """
				if not test:
					""" doesn't take into account words that occur in test-set but didn't occur in traning-set"""
					word_position = self.next_word_position
					self.next_word_position += 1
					tweet_words_dict[word_position] = 1
					self.words_dict[token] = word_position

		return token_morpho_features, token_vectors

	@staticmethod
	def prod(iterable):
		return reduce(operator.mul, iterable, 1)

	def analyze_token_list(self, tokens, test, pos_list=None):
		"""
			Analyzes the list of tokens and the list of token pos

			:param tokens: list of tokens
			:param test: boolean, true if the document is the test set, false otherwise
			:param pos_list: list of tokens pos
			:return: feature vector (sample) and dictionary of word occurrences
		"""
		tweet_length = len(tokens)
		sample = []
		""" assign function to variable """
		push_to_sample = sample.extend
		"""
			empty dictionary for word occurrences;
			keys are indexes assigned to words in overall dictionary;
			values are occurrences of words in tweet;
			this dictionary will be updated each time a word of the tweet is analyzed
		"""
		tweet_words_dict = {}
		""" dictionary with features based on words and emojs """
		tweet_morpho_features = {
			'caps_count': 0,
			'non_caps_count': 0,
			'pos_emoj_count': 0,
			'neg_emoj_count': 0,
			'elong_words_count': 0,
			'negations_count': 0
		}
		pos_all = self.properties['POS_ALL']
		"""
			creation of vector of dictionaries for current tweet's embeddings;
			each dictionary in the vector will store the embeddings for one distributional model;
			in each dictionary:
				- the key is the PoS selection
				- the value is the list of the embeddings obtained with that PoS selection
		"""
		if pos_all:
			vector_dicts = [{'all': []}] * len(self.vector_models)
		else:
			vector_dicts = [{}] * len(self.vector_models)
		use_pos = len(self.properties['POS_LIST']) > 0
		if use_pos:
			for k in range(len(self.vector_models)):
				for pos in self.properties['POS_LIST']:
					# for each chosen PoS, an empty list is added
					vector_dicts[k][pos] = []
		for i in range(tweet_length):
			pos = pos_list[i]
			""" 
				for the current token, we obtain:
					- the features based on word-shape or on the emoj type
					- the vectors obtained combining the embeddings
			"""
			token_morpho_features, token_vectors = self.analyze_token(tokens[i], tweet_words_dict, test)
			for key, value in token_morpho_features.iteritems():
				""" word and emoj features of the tweet are updated with word and emoj features of the token """
				tweet_morpho_features[key] += value
			for j, vector in enumerate(token_vectors):
				"""
					token_vectors contains an embedding for each distributional model we use
				"""
				if vector is not None:
					if pos_all:
						""" if we use all word, we add the embedding of current word """
						vector_dicts[j]['all'].append(vector)
					if use_pos:
						""" if the pos of the word is in the list of chosen pos, we add the embedding of current word"""
						if pos in self.properties['POS_LIST']:
							vector_dicts[j][pos].append(vector)
		""" the feature vector is updated with word-based and emoj-based features """
		if self.properties['ALL_CAPS']:
			push_to_sample([int(tweet_morpho_features['non_caps_count'] == 0 and tweet_morpho_features['caps_count'] > 0)])
			self.allCapTweets += int(tweet_morpho_features['non_caps_count'] == 0 and tweet_morpho_features['caps_count'] > 0)
		if self.properties['CONTAINS_CAPS']:
			push_to_sample([int(tweet_morpho_features['caps_count'] > 0)])
			self.containsCaps += int(tweet_morpho_features['caps_count'] > 0)
		if self.properties['POSITIVE_EMOTICONS']:
			push_to_sample([int(tweet_morpho_features['pos_emoj_count'] > 0)])
			self.positiveEmo += int(tweet_morpho_features['pos_emoj_count'] > 0)
		if self.properties['NEGATIVE_EMOTICONS']:
			push_to_sample([int(tweet_morpho_features['neg_emoj_count'] > 0)])
			self.negativeEmo += int(tweet_morpho_features['neg_emoj_count'] > 0)
		if self.properties['ELONGATED_WORDS']:
			push_to_sample([int(tweet_morpho_features['elong_words_count'] > 0)])
			self.elongated += int(tweet_morpho_features['elong_words_count'] > 0)
		if self.properties['NEGATIONS']:
			push_to_sample([int(tweet_morpho_features['negations_count'] > 0)])
			self.negations += int(tweet_morpho_features['negations_count'] > 0)
		use_sum = self.properties['SUM_VECTOR']
		use_avg = self.properties['AVERAGE_VECTOR']
		use_prod = self.properties['PRODUCT_VECTOR']
		use_min = self.properties['MIN_POOL_VECTOR']
		use_max = self.properties['MAX_POOL_VECTOR']
		if self.properties['USE_WORD_VECTORS']:
			for vector_dict in vector_dicts:
				""" for each pos selection we use the chosen combination methods """
				for key in self.properties['POS_LIST']:
					vector_list = vector_dict[key]
					try:
						assert len(vector_list) > 0
						# vettore con i numeri zippati per fare i calcoli sulle componenti
						values_vector = zip(*vector_list)
						if use_sum:
							sum_vector = map(sum, values_vector)
							push_to_sample(sum_vector)
							if use_avg:
								push_to_sample([x/tweet_length for x in sum_vector])
						if use_avg and not use_sum:
							sum_vector = map(sum, values_vector)
							push_to_sample([x/tweet_length for x in sum_vector])
						if use_prod:
							push_to_sample(map(self.prod, values_vector))
						if use_min:
							push_to_sample(map(min, values_vector))
						if use_max:
							push_to_sample(map(max, values_vector))
					except AssertionError:
						""" in this case no embeddings has been found for given pos selection """
						if use_sum:
							push_to_sample(self.zero_vector)
						if use_avg:
							push_to_sample(self.zero_vector)
						if use_prod:
							push_to_sample(self.zero_vector)
						if use_min:
							push_to_sample(self.zero_vector)
						if use_max:
							push_to_sample(self.zero_vector)
		return sample, tweet_words_dict

	def extract_from_conll(self, filename, test=False):
		"""
			Main function of the class
			:param filename: name of the file for which we extract the features
			:param test: boolean, true if this is the test file, false otherwise
			:return:
		"""
		samples = [] # list of the feature vectors
		tweet_words_dict_s = [] # list of dictionaries, each contains the occurrences of words in the document (tweet)
		subj_s = [] # list of labels for Subjectivity Classification task
		opos_s = [] # list of labels for overall positivity
		oneg_s = [] # list of labels for overall negativity
		iro_s = [] # list of labels for Irony Detection task
		lpos_s = [] # list of labels for literal positivity
		lneg_s = [] # list of labels for literal negativity
		id_s = [] # list of tweets ids
		top_s = [] # list of topic labels

		""" assigning functions to variables, because they're going to be used often """
		add_sample = samples.append
		add_words_dict = tweet_words_dict_s.append
		add_subj = subj_s.append
		add_opos = opos_s.append
		add_oneg = oneg_s.append
		add_iro = iro_s.append
		add_lpos = lpos_s.append
		add_lneg = lneg_s.append
		add_id = id_s.append
		add_top = top_s.append
		with codecs.open(filename, 'r', 'utf8') as infile:
			""" Reading the file """
			tokens = [] # temporary variable for each tweet's tokens
			pos_list = [] # temporary variable for each tweet's tokens pos
			pattern = re.compile('"\d+"') # regex pattern to find the labels
			for line in infile:
				""" assigning functions to variables, because they're going to be used repeatedly"""
				add_token = tokens.append
				add_pos = pos_list.append
				if '<doc' in line: # Start of a document: labels are added to the corresponding lists
					labels = pattern.findall(line)
					add_id(int(labels[0].strip('"')))
					add_subj(int(labels[1].strip('"')))
					add_opos(int(labels[2].strip('"')))
					add_oneg(int(labels[3].strip('"')))
					add_iro(int(labels[4].strip('"')))
					add_lpos(int(labels[5].strip('"')))
					add_lneg(int(labels[6].strip('"')))
					add_top(int(labels[7].strip('"')))
				elif line[0].isdigit(): # lines containing tokens
					fields = line.split('\t')
					token = fields[1]
					pos = fields[3]
					add_token(token) # token is added to list of tokens of the tweet
					add_pos(pos) # pos is added to list of tokens pos of the tweet
				elif '</doc>' in line: # end of a document
					"""
						the token list and the pos list are analyzed
						we obtain (for the current tweet) the feature vector (sample) and the dictionary of word occurrences
					"""
					sample, tweet_words_dict = self.analyze_token_list(tokens, test, pos_list=pos_list)
					"""
						the feature vector and the dictionary of word occurrences are added to the corresponding lists
						the temporary lists are emptied
					"""
					add_sample(sample)
					add_words_dict(tweet_words_dict)
					tokens = []
					pos_list = []
		if test:
			""" if the analyzed document is the test, we keep the tweets ids and topics """
			return samples, tweet_words_dict_s, id_s, top_s, subj_s, opos_s, oneg_s, iro_s, lpos_s, lneg_s
		else:
			return samples, tweet_words_dict_s, subj_s, opos_s, oneg_s, iro_s, lpos_s, lneg_s