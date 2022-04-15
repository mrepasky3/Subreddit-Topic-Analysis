import os
import argparse
import pandas as pd
import numpy as np
from scipy import signal
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates
from wordcloud import WordCloud
from sklearn.linear_model import TheilSenRegressor
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, Coherence, CoherenceCV
from utils import *

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import gensim
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

parser = argparse.ArgumentParser()
parser.add_argument('--n_topics', type=int, default=15)
parser.add_argument('--weekly_series', action="store_true")
parser.add_argument('--daily_series', action="store_true")
parser.add_argument('--load', action="store_true")
parser.add_argument('--ctm_load', action="store_true")
parser.set_defaults(feature=False)


if __name__ == '__main__':
	args = parser.parse_args()
	if not os.path.exists("results"):
		os.mkdir('results')
	
	stop_words = stopwords.words('english')
	stop_words.extend(['im','ive','dont','get','youre','would','thats',
		'really','one','also','something','even','thing','things','must',
		'cant','much','could','way','lot','got','get','go','like','th'])

	comments_list = []
	for data_file in os.listdir('weekly_data'):
		loaded_comments = pd.read_csv('weekly_data/' + data_file)
		comments_list += list(loaded_comments['body'])

	comments_words, bigram_model, trigram_model = tokenize_lda(data_cleaning(comments_list))
	comments_bow, dictionary = create_dictionary(comments_words)

	if args.ctm_load:
		documents = [line.strip() for line in comments_list]
		sp = WhiteSpacePreprocessing(documents, stopwords_language='english')
		preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()

		tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v1")

		training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

		print("Fitting model")
		ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=15, num_epochs=10)
		ctm.fit(training_dataset) # run the model

		#ctm.save(models_dir="results/CTM_Model/")

		texts = [doc.split() for doc in preprocessed_documents]

		cv = CoherenceModel(topics=ctm.get_topic_lists(20), texts=texts, dictionary=Dictionary(texts), coherence='c_v', topn=20)

		per_topic_coherence = cv.get_coherence_per_topic()

		topics_sorted_by_coherence = np.argsort(per_topic_coherence)[::-1]
		topics_sorted_by_coherence = list(topics_sorted_by_coherence[:9])

		stop_words = stopwords.words('english')
		stop_words.extend(['im','ive','dont','get','youre','would','thats',
				'really','one','also','something','even','thing','things','must',
				'cant','much','could','way','lot','got','get','go','like','th'])

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
		plt.savefig("CTM_Word_Cloud.png")
		plt.close()
		plt.clf()
