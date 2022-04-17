import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import *

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import gensim
import gensim.models.nmf
from gensim.models import CoherenceModel




parser = argparse.ArgumentParser()
parser.add_argument('--n_topics', type=int, default=15)
parser.set_defaults(feature=False)




if __name__ == '__main__':
	args = parser.parse_args()
	if not os.path.exists("results"):
		os.mkdir('results')
	elif not os.path.exists("results/NMF_Model/"):
		os.mkdir('results/NMF_Model/')
	
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

	nmf = gensim.models.nmf.Nmf(corpus=comments_bow, num_topics=args.n_topics, id2word=dictionary,
		w_max_iter=300, h_max_iter=100)
	nmf.save("results/NMF_Model/NMF_model")

	# get coherence
	coherence_model = CoherenceModel(model=nmf, texts=comments_words, dictionary=dictionary, coherence='c_v')
	print("NMF Coherence: {:.3f}".format(coherence_model.get_coherence()))
	print("NMF Per-Topic Coherence: "+str(coherence_model.get_coherence_per_topic()))

	report_file = open("results/NMF_Model/nmf_coherence.txt", "w")
	report_file.write("NMF Coherence: {:.3f}\n".format(coherence_model.get_coherence()))
	per_topic_coherence = coherence_model.get_coherence_per_topic()
	for i in range(len(per_topic_coherence)):
		report_file.write("Topic {} Coherence: {:.3f}\n".format(i, per_topic_coherence[i]))
	report_file.close()

	topics_sorted_by_coherence = np.argsort(per_topic_coherence)[::-1]
	topics_sorted_by_coherence = list(topics_sorted_by_coherence[:9])

	print(topics_sorted_by_coherence)

	topics = nmf.show_topics(num_topics=args.n_topics, formatted=False, num_words=20)
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
	plt.savefig("results/NMF_Word_Cloud.png")
	plt.close()
	plt.clf()