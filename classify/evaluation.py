#!/usr/bin/env python
# evaluation script for the SENTIPOLC 2016 shared task


def evaluate(result_file, gold_file, verbose=True, outfile=''):
	output_string = ''
	# reads the gold standard and populate the count matrix
	gold = dict()
	gold_counts = {
		'subj': {'0': 0.0, '1': 0.0},
		'opos': {'0': 0.0, '1': 0.0},
		'oneg': {'0': 0.0, '1': 0.0},
		'iro': {'0': 0.0, '1': 0.0},
		'lpos': {'0': 0.0, '1': 0.0},
		'lneg': {'0': 0.0, '1': 0.0}
	}
	with open(gold_file) as f:
		for line in f:
			try:
				assert len(line) > 1
			except AssertionError:
				continue
			if len(line.split(',')) > 6:
				id, subj, opos, oneg, iro, lpos, lneg, top = map(lambda x: x[1:-1], line.rstrip().split(','))
				gold_counts['lpos'][lpos] += 1
				gold_counts['lneg'][lneg] += 1

			else:
				id, subj, opos, oneg, iro, top = map(lambda x: x[1:-1], line.rstrip().split(','))
			gold[id] = {'subj': subj, 'opos': opos, 'oneg': oneg, 'iro': iro, 'lpos': lpos, 'lneg': lneg}
			gold_counts['subj'][subj] += 1
			gold_counts['opos'][opos] += 1
			gold_counts['oneg'][oneg] += 1
			gold_counts['iro'][iro] += 1

	# reads the result data
	result = dict()
	with open(result_file) as f:
		for line in f:
			try:
				assert len(line) > 1
			except AssertionError:
				continue
			if len(line.split(',')) > 6:
				id, subj, opos, oneg, iro, lpos, lneg, top = map(lambda x: x[1:-1], line.rstrip().split(','))
				result[id] = {
					'subj': subj,
					'opos': opos,
					'oneg': oneg,
					'iro': iro
				}
			else:
				id, subj, opos, oneg, iro, top = map(lambda x: x[1:-1], line.rstrip().split(','))
				result[id] = {'subj': subj, 'opos': opos, 'oneg': oneg, 'iro': iro}

	# evaluation: single classes
	for task in ['subj', 'opos', 'oneg', 'iro']:  # add 'lpos' and 'lneg' if you want to measure literal polairty
		# table header
		if verbose:
			print "\ntask: {}".format(task)
		else:
			output_string += "\ntask: {}".format(task)
			output_string += '\n'
		if verbose:
			print "prec. 0\trec. 0\tF-sc. 0\tprec. 1\trec. 1\tF-sc. 1\tF-sc."
		else:
			output_string += "prec. 0\trec. 0\tF-sc. 0\tprec. 1\trec. 1\tF-sc. 1\tF-sc."
			output_string += '\n'
		correct = {'0': 0.0, '1': 0.0}
		assigned = {'0': 0.0, '1': 0.0}
		precision = {'0': 0.0, '1': 0.0}
		recall = {'0': 0.0, '1': 0.0}
		fscore = {'0': 0.0, '1': 0.0}

		# counts the labels
		for id, gold_labels in gold.iteritems():
			if (not id in result) or result[id][task] == '':
				pass
			else:
				assigned[result[id][task]] += 1
				if gold_labels[task] == result[id][task]:
					correct[result[id][task]] += 1

		# computes precision, recall and F-score
		for label in ['0', '1']:
			try:
				precision[label] = float(correct[label]) / float(assigned[label])
				recall[label] = float(correct[label]) / float(gold_counts[task][label])
				fscore[label] = (2.0 * precision[label] * recall[label]) / (precision[label] + recall[label])
			except:
				# if a team doesn't participate in a task it gets default 0 F-score
				fscore[label] = 0.0

		# writes the table
		if verbose:
			print "{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}".format(
				precision['0'],
				recall['0'],
				fscore['0'],
				precision['1'],
				recall['1'],
				fscore['1'],
				(fscore['0'] + fscore['1']) / 2.0
			)
		else:
			output_string += "{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}".format(
				precision['0'],
				recall['0'],
				fscore['0'],
				precision['1'],
				recall['1'],
				fscore['1'],
				(fscore['0'] + fscore['1']) / 2.0
			)
			output_string += '\n'

	# polarity evaluation needs a further step
	if verbose:
		print "\ntask: polarity"
	else:
		output_string += "\ntask: polarity"
		output_string += '\n'
	if verbose:
		print "Combined F-score"
	else:
		output_string += "Combined F-score"
		output_string += '\n'
	correct = {
		'opos': {'0': 0.0, '1': 0.0},
		'oneg': {'0': 0.0, '1': 0.0}
	}
	assigned = {
		'opos': {'0': 0.0, '1': 0.0},
		'oneg': {'0': 0.0, '1': 0.0}
	}
	precision = {
		'opos': {'0': 0.0, '1': 0.0},
		'oneg': {'0': 0.0, '1': 0.0}
	}
	recall = {
		'opos': {'0': 0.0, '1': 0.0},
		'oneg': {'0': 0.0, '1': 0.0}
	}
	fscore = {
		'opos': {'0': 0.0, '1': 0.0},
		'oneg': {'0': 0.0, '1': 0.0}
	}

	# counts the labels
	for id, gold_labels in gold.iteritems():
		for cl in ['opos', 'oneg']:
			if (not id in result) or result[id][cl] == '':
				pass
			else:
				assigned[cl][result[id][cl]] += 1
				if gold_labels[cl] == result[id][cl]:
					correct[cl][result[id][cl]] += 1

	# computes precision, recall and F-score
	for cl in ['opos', 'oneg']:
		for label in ['0', '1']:
			try:
				precision[cl][label] = float(correct[cl][label]) / float(assigned[cl][label])
				recall[cl][label] = float(correct[cl][label]) / float(gold_counts[cl][label])
				x = float(2.0 * precision[cl][label] * recall[cl][label])
				y = float(precision[cl][label] + recall[cl][label])
				fscore[cl][label] = x/y
			
			except:
				fscore[cl][label] = 0.0

	fscore_pos = (fscore['opos']['0'] + fscore['opos']['1']) / 2.0
	fscore_neg = (fscore['oneg']['0'] + fscore['oneg']['1']) / 2.0

	# writes the table
	if verbose:
		print "{0:.4f}".format((fscore_pos + fscore_neg) / 2.0)
	else:
		output_string += "{0:.4f}".format((fscore_pos + fscore_neg) / 2.0)
		output_string += '\n'
	outfile.write(output_string)


if __name__ == '__main__':

	from argparse import ArgumentParser

	argparser = ArgumentParser(description='')
	argparser.add_argument(
		'-r',
		dest='result_file',
		action='store',
		help='CSV file of the run results'
	)
	argparser.add_argument(
		'-g',
		dest='gold_file',
		action='store',
		default="sentipolc16_gold2000.csv",
		help='gold standard annotation CSV file'
	)
	args = argparser.parse_args()
	evaluate(args.result_file, args.gold_file)
