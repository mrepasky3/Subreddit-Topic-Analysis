import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import *

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import gensim
from gensim.models import CoherenceModel




if __name__ == "__main__":
	'''
	Generate Word Cloud for LDA topics which are not
	in the top 9 most coherent topics.
	'''
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

	lda = gensim.models.LdaModel.load("results/LDA_model")

	coherence_model = CoherenceModel(model=lda, texts=comments_words, dictionary=dictionary, coherence='c_v')
	per_topic_coherence = coherence_model.get_coherence_per_topic()
	topics_sorted_by_coherence = np.argsort(per_topic_coherence)[::-1]
	topics_sorted_by_coherence = list(topics_sorted_by_coherence[9:])

	topics = lda.show_topics(num_topics=15, formatted=False, num_words=20)
	fig, axs = plt.subplots(2, 3, figsize=(13,8))
	cloud = WordCloud(stopwords=stop_words,background_color='white')
	k=0
	for i in range(2):
		for j in range(3):
			while k not in topics_sorted_by_coherence:
				k+=1
			print(k)
			for topic in topics:
				if topic[0] == k:
					topic_dict = dict(topic[1])
			cloud.generate_from_frequencies(topic_dict, max_font_size=300)
			axs[i,j].imshow(cloud)
			axs[i,j].set_title('Topic ' + str(k) + r' | $C_v$: ' + '{:.2f}'.format(per_topic_coherence[k]), fontsize=26)
			axs[i,j].axis('off')
			k+=1
	plt.tight_layout()
	plt.savefig("results/LDA_Word_Cloud_worsetopics.png")
	plt.close()
	plt.clf()