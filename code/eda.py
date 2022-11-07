# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
random.seed(1)

#stop words list
#stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
#			'ours', 'ourselves', 'you', 'your', 'yours', 
#			'yourself', 'yourselves', 'he', 'him', 'his', 
#			'himself', 'she', 'her', 'hers', 'herself', 
#			'it', 'its', 'itself', 'they', 'them', 'their', 
#			'theirs', 'themselves', 'what', 'which', 'who', 
#			'whom', 'this', 'that', 'these', 'those', 'am', 
#			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
#			'have', 'has', 'had', 'having', 'do', 'does', 'did',
#			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
#			'because', 'as', 'until', 'while', 'of', 'at', 
#			'by', 'for', 'with', 'about', 'against', 'between',
#			'into', 'through', 'during', 'before', 'after', 
#			'above', 'below', 'to', 'from', 'up', 'down', 'in',
#			'out', 'on', 'off', 'over', 'under', 'again', 
#			'further', 'then', 'once', 'here', 'there', 'when', 
#			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
#			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
#			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
#			'very', 's', 't', 'can', 'will', 'just', 'don', 
#			'should', 'now', '']

#cleaning up text
import re

def get_only_hangul(line):
	#print(line)
	parseText= re.compile('/ ^[ㄱ-ㅎㅏ-ㅣ가-힣]*$/').sub('',line)

	return parseText

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.sejong import ssem
from nltk import nouns

def synonym_replacement(words, n):
	new_words = words.copy()
	#print("new_words : %s" % new_words)
	
	random_word_list = list(set([word for word in words]))

	#print(random_word_list)
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		entrys = ssem.entrys(random_word)
		if (len(entrys) != 0):
			synonyms = get_synonyms(random_word)
			if len(synonyms) >= 1:
				synonym = random.choice(list(synonyms))
				new_words = [synonym if word == random_word else word for word in new_words]
				#print("replaced", random_word, "with", synonym)
				num_replaced += 1
			if num_replaced >= n:
				break
		else:
			synonyms = random_word


	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def get_synonyms(word):
	#print(word)
	entrys = ssem.entrys(word)
	#print(entrys)
	sense = entrys[0].senses()[0]
	#print(sense)
	syn = sense.syn()
	if len(syn) !=0:
		return syn
	else:
		return word

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	return new_words

def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		if len(new_words) >= 1:
			random_word = new_words[random.randint(0, len(new_words)-1)]
			entrys = ssem.entrys(random_word)
			if (len(entrys) != 0):
				synonyms = get_synonyms(random_word)
				counter += 1
			else:
				synonyms = random_word
				counter +=1
		else:
			random_word = ""

		if counter >= 10:
			return
		
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)


########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
	
	sentence = get_only_hangul(sentence)
	#print(sentence)
	words = sentence.split(' ')
	#words = nouns(sentence)
	#print(words)
	words = [word for word in words if word is not '']
	num_words = len(words)
	#print(num_words)

	#augmented_sentence = []
	augmented_sentences = []
	num_new_per_technique = int(num_aug)+1
	#print(num_new_per_technique)

	#sr
	if (alpha_sr > 0):
		n_sr = max(1, int(alpha_sr*num_words))
		#print(n_sr)
		for _ in range(num_new_per_technique):
			a_words = synonym_replacement(words, n_sr)
			#print(a_words)
			#for a_word in a_words:
			#	augmented_sentence.append(' '.join(a_word))
			#	print(augmented_sentence)
			#augmented_sentences.append(augmented_sentences)
			augmented_sentences.append(' '.join(a_words))

	#ri
	if (alpha_ri > 0):
		n_ri = max(1, int(alpha_ri*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri)
			augmented_sentences.append(' '.join(a_words))

	#rs
	if (alpha_rs > 0):
		n_rs = max(1, int(alpha_rs*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs)
			augmented_sentences.append(' '.join(a_words))

	#rd
	if (p_rd > 0):
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, p_rd)
			augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_hangul(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	augmented_sentences.append(sentence)

	return augmented_sentences