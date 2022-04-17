import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import *

import gensim
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords

import pickle

from gensim.topic_coherence import direct_confirmation_measure
from coherence_fix import custom_log_ratio_measure
direct_confirmation_measure.log_ratio_measure = custom_log_ratio_measure

parser = argparse.ArgumentParser()
parser.add_argument('--n_topics', type=int, default=15)
parser.add_argument('--topic_model', choices=['lda', 'nmf', 'ctm'])
parser.set_defaults(feature=False)


if __name__ == '__main__':
	args = parser.parse_args()

	comments_list = []
	for data_file in os.listdir('weekly_data'):
		loaded_comments = pd.read_csv('weekly_data/' + data_file)
		comments_list += list(loaded_comments['body'])


	if args.topic_model == 'lda':
		if not os.path.exists("results/lda_clouds"):
			os.mkdir('results/lda_clouds')
		report_file = open("results/lda_clouds/coherence.txt", 'w')

		stop_words = stopwords.words('english')
		stop_words.extend(['im','ive','dont','get','youre','would','thats',
			'really','one','also','something','even','thing','things','must',
			'cant','much','could','way','lot','got','get','go','like','th'])

		comments_words, bigram_model, trigram_model = tokenize_lda(data_cleaning(comments_list))
		comments_bow, dictionary = create_dictionary(comments_words)

		lda = gensim.models.LdaModel.load("results/LDA_model")

		cv = CoherenceModel(model=lda, texts=comments_words, dictionary=dictionary, coherence='c_v')
		report_file.write("Overall Coherence: {:.3f}\n".format(cv.get_coherence()))

		per_topic_coherence = cv.get_coherence_per_topic()
		topics_sorted_by_coherence = np.argsort(per_topic_coherence)[::-1]
		topics_sorted_by_coherence = list(topics_sorted_by_coherence)

		topics = lda.show_topics(num_topics=args.n_topics, formatted=False, num_words=20)

		cloud = WordCloud(stopwords=stop_words,background_color='white',width=800, height=400)
		for k in range(args.n_topics):
			fig = plt.figure(figsize=(20,10))
			for topic in topics:
				if topic[0] == k:
					topic_dict = dict(topic[1])
			report_file.write('Topic {} C_v: {:.3f}\n'.format(k, per_topic_coherence[k]))
			cloud.generate_from_frequencies(topic_dict, max_font_size=300)
			plt.imshow(cloud)
			plt.axis('off')
			plt.savefig("results/lda_clouds/lda_topic{}.png".format(k))
			plt.close()
			plt.clf()

		report_file.close()


	elif args.topic_model == 'nmf':
		if not os.path.exists("results/nmf_clouds"):
			os.mkdir('results/nmf_clouds')
		report_file = open("results/nmf_clouds/coherence.txt", 'w')

		stop_words = stopwords.words('english')
		stop_words.extend(['im','ive','dont','get','youre','would','thats',
			'really','one','also','something','even','thing','things','must',
			'cant','much','could','way','lot','got','get','go','like','th'])

		comments_words, bigram_model, trigram_model = tokenize_lda(data_cleaning(comments_list))
		comments_bow, dictionary = create_dictionary(comments_words)

		lda = gensim.models.LdaModel.load("results/NMF_Model/NMF_model")

		cv = CoherenceModel(model=lda, texts=comments_words, dictionary=dictionary, coherence='c_v')
		report_file.write("Overall Coherence: {:.3f}\n".format(cv.get_coherence()))

		per_topic_coherence = cv.get_coherence_per_topic()
		topics_sorted_by_coherence = np.argsort(per_topic_coherence)[::-1]
		topics_sorted_by_coherence = list(topics_sorted_by_coherence)

		topics = lda.show_topics(num_topics=args.n_topics, formatted=False, num_words=20)

		cloud = WordCloud(stopwords=stop_words,background_color='white',width=800, height=400)
		for k in range(args.n_topics):
			fig = plt.figure(figsize=(20,10))
			for topic in topics:
				if topic[0] == k:
					topic_dict = dict(topic[1])
			report_file.write('Topic {} C_v: {:.3f}\n'.format(k, per_topic_coherence[k]))
			cloud.generate_from_frequencies(topic_dict, max_font_size=300)
			plt.imshow(cloud)
			plt.axis('off')
			plt.savefig("results/nmf_clouds/nmf_topic{}.png".format(k))
			plt.close()
			plt.clf()

		report_file.close()


	elif args.topic_model == 'ctm':
		if not os.path.exists("results/ctm_clouds"):
			os.mkdir('results/ctm_clouds')
		report_file = open("results/ctm_clouds/coherence.txt", 'w')

		batch_size = 64
		documents = [line.strip() for line in comments_list]
		documents = documents[: len(documents) // batch_size * batch_size]
		sp = WhiteSpacePreprocessing(stopwords_language='english')
		preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess(documents)

		with open("results/CTM_Model/training_dataset.pkl", "rb") as f:
			training_dataset = pickle.load(f)
		with open("results/CTM_Model/ctm.pkl", "rb") as f:
				ctm = pickle.load(f)

		texts = [doc.split() for doc in preprocessed_documents]

		dictionary = Dictionary()
		for key in training_dataset.idx2token.keys():
			dictionary.add_documents([[training_dataset.idx2token[key]]])

		cv = CoherenceModel(topics=ctm.get_topic_lists(20), texts=texts, dictionary=dictionary, coherence='c_v', topn=20)
		report_file.write("Overall Coherence: {:.3f}\n".format(cv.get_coherence()))

		per_topic_coherence = cv.get_coherence_per_topic()
		topics_sorted_by_coherence = np.argsort(per_topic_coherence)[::-1]
		topics_sorted_by_coherence = list(topics_sorted_by_coherence)

		stop_words = stopwords.words('english')

		topic_words = ctm.get_topics(20)
		topics = [(i, ctm.get_word_distribution_by_topic_id(i)[:20]) for i in range(len(topic_words))]

		cloud = WordCloud(stopwords=stop_words,background_color='white',width=800, height=400)
		for k in range(args.n_topics):
			fig = plt.figure(figsize=(20,10))
			for topic in topics:
				if topic[0] == k:
					topic_dict = dict(topic[1])
			report_file.write('Topic {} C_v: {:.3f}\n'.format(k, per_topic_coherence[k]))
			cloud.generate_from_frequencies(topic_dict, max_font_size=300)
			plt.imshow(cloud)
			plt.axis('off')
			plt.savefig("results/ctm_clouds/ctm_topic{}.png".format(k))
			plt.close()
			plt.clf()

		report_file.close()