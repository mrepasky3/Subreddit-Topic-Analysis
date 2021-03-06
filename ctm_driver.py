import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.evaluation.measures import CoherenceCV
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
parser.add_argument('--ctm_load', action="store_true")
parser.add_argument('--tmdp_load', action="store_true")
parser.set_defaults(feature=False)


if __name__ == '__main__':
	args = parser.parse_args()
	if not os.path.exists("results"):
		os.mkdir('results')

	comments_list = []
	for data_file in os.listdir('weekly_data'):
		loaded_comments = pd.read_csv('weekly_data/' + data_file)
		comments_list += list(loaded_comments['body'])

	batch_size = 64
	documents = [line.strip() for line in comments_list]
	documents = documents[: len(documents) // batch_size * batch_size]
	sp = WhiteSpacePreprocessing(stopwords_language='english')
	preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess(documents)

	if args.tmdp_load:
		with open("results/CTM_Model/training_dataset.pkl", "rb") as f:
			training_dataset = pickle.load(f)
		with open("results/CTM_Model/tmdp.pkl", "rb") as f:
			tp = pickle.load(f)
	else:
		tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v1")
		training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
		with open("results/CTM_Model/training_dataset.pkl", "wb") as f:
			pickle.dump(training_dataset, f)
		with open("results/CTM_Model/tmdp.pkl", "wb") as f:
			pickle.dump(tp, f)

	if args.ctm_load:
		with open("results/CTM_Model/ctm.pkl", "rb") as f:
  			ctm = pickle.load(f)
	else:
		print("Fitting model")
		ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=15, num_epochs=10, batch_size=batch_size)
		ctm.fit(training_dataset) # run the model
		with open("results/CTM_Model/ctm.pkl", "wb") as f:
			pickle.dump(ctm, f)

	texts = [doc.split() for doc in preprocessed_documents]

	dictionary = Dictionary()
	for key in training_dataset.idx2token.keys():
		dictionary.add_documents([[training_dataset.idx2token[key]]])

	cv = CoherenceModel(topics=ctm.get_topic_lists(20), texts=texts, dictionary=dictionary, coherence='c_v', topn=20)

	per_topic_coherence = cv.get_coherence_per_topic()

	print(per_topic_coherence, cv.get_coherence())

	topics_sorted_by_coherence = np.argsort(per_topic_coherence)[::-1]
	topics_sorted_by_coherence = list(topics_sorted_by_coherence[:9])

	stop_words = stopwords.words('english')

	topic_words = ctm.get_topics(20)
	topics = [(i, ctm.get_word_distribution_by_topic_id(i)[:20]) for i in range(len(topic_words))]
	topic_dict = dict()
	fig, axs = plt.subplots(3, 3, figsize=(20,8))
	cloud = WordCloud(stopwords=stop_words,background_color='white')
	k=0
	for i in range(3):
	  for j in range(3):
	    while k not in topics_sorted_by_coherence:
	      k+=1
	    for topic in topics:
	      if topic[0] == k:
	        topic_dict = dict(topic[1])
	    cloud.generate_from_frequencies(topic_dict, max_font_size=300)
	    axs[i,j].imshow(cloud)
	    axs[i,j].set_title('Topic ' + str(k) + r' | $C_v$: ' + '{:.2f}'.format(per_topic_coherence[k]), fontsize=26)
	    axs[i,j].axis('off')
	    k+=1
	plt.tight_layout()
	plt.savefig("results/CTM_Word_Cloud.png")
	plt.close()
	plt.clf()