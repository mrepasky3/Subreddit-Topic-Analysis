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
from utils import *

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import gensim
from gensim.models import CoherenceModel




parser = argparse.ArgumentParser()
parser.add_argument('--n_topics', type=int, default=15)
parser.add_argument('--weekly_series', action="store_true")
parser.add_argument('--daily_series', action="store_true")
parser.add_argument('--load', action="store_true")
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

	if not args.load:
		lda = gensim.models.LdaMulticore(corpus=comments_bow, id2word=dictionary, num_topics=args.n_topics,
								   chunksize=500, passes=10, alpha='asymmetric', eta='symmetric')
		lda.save("results/LDA_model")

		coherence_model = CoherenceModel(model=lda, texts=comments_words, dictionary=dictionary, coherence='c_v')
		print("LDA Coherence: {:.3f}".format(coherence_model.get_coherence()))
		print("LDA Per-Topic Coherence: "+str(coherence_model.get_coherence_per_topic()))

		report_file = open("results/lda_coherence.txt", "w")
		report_file.write("LDA Coherence: {:.3f}\n".format(coherence_model.get_coherence()))
		per_topic_coherence = coherence_model.get_coherence_per_topic()
		for i in range(len(per_topic_coherence)):
			report_file.write("Topic {} Coherence: {:.3f}\n".format(i, per_topic_coherence[i]))
		report_file.close()

		topics_sorted_by_coherence = np.argsort(per_topic_coherence)[::-1]
		topics_sorted_by_coherence = list(topics_sorted_by_coherence[:9])

		print(topics_sorted_by_coherence)

		topics = lda.show_topics(num_topics=args.n_topics, formatted=False, num_words=20)
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
		plt.savefig("results/LDA_Word_Cloud.png")
		plt.close()
		plt.clf()

	else:
		lda = gensim.models.LdaModel.load("results/LDA_model")

		coherence_model = CoherenceModel(model=lda, texts=comments_words, dictionary=dictionary, coherence='c_v')
		per_topic_coherence = coherence_model.get_coherence_per_topic()
		topics_sorted_by_coherence = np.argsort(per_topic_coherence)[::-1]
		topics_sorted_by_coherence = list(topics_sorted_by_coherence[:9])




	if args.weekly_series:
		'''
		For each week in the data, collect comments, tokenize them, and form
		BoW vectors based on pre-defined dictionary. Then, calculate how many times
		each topic is the predominant topic of a comment for each week. Plot this as
		a time series, plot the periodogram (reporting top five periods), and plot
		the Theil-Sen slope estimate.
		'''

		# create dictionary of tokenized comment lists for each week
		comments_list_weekly = dict()
		for data_file in os.listdir('weekly_data'):
			start = data_file.split('-')[0][-10:]
			start_time = int(dt.datetime.strptime(start, '%d_%m_%Y').timestamp())
			
			loaded_comments = list(pd.read_csv('weekly_data/' + data_file)['body'])
			tokenized_loaded_comments, _, _ = tokenize_lda(data_cleaning(loaded_comments), bigram_model=bigram_model, trigram_model=trigram_model)
			bow_loaded_comments = [dictionary.doc2bow(comment) for comment in tokenized_loaded_comments]
			comments_list_weekly[start_time] = bow_loaded_comments
		
		# create dictionary tracking how frequently each topic is present per week
		topic_multiplicity_weekly = dict()
		for key in comments_list_weekly.keys():
			topic_multiplicity_weekly[key] = np.zeros(args.n_topics)
			this_week_comments = comments_list_weekly[key]
			for comment in this_week_comments:
				topic_dist = lda.get_document_topics(comment,per_word_topics=False)
				top_topic_idx = np.argmax(np.array(topic_dist)[:,1])
				top_topic = topic_dist[top_topic_idx][0]
				topic_multiplicity_weekly[key][top_topic] += 1

		# transform this dictionary into a sorted array
		topic_trends = []
		for key in topic_multiplicity_weekly.keys():
			topic_trends.append([key]+list(topic_multiplicity_weekly[key]))
		topic_trends = np.array(topic_trends)
		sorted_idx = np.argsort(topic_trends[:,0])
		sorted_topic_trends = topic_trends[sorted_idx]
		topic_trend_dates = [dt.datetime.fromtimestamp(stamp) for stamp in sorted_topic_trends[:,0]]

		color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',
					  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

		fig, axs = plt.subplots(3, 3, figsize=(24,15))
		k = 0
		color_counter = 0
		for i in range(3):
			for j in range(3):
				while k not in topics_sorted_by_coherence:
					k+=1
				axs[i,j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
				axs[i,j].scatter(topic_trend_dates, sorted_topic_trends[:,k+1], marker='o', color=color_list[color_counter])
				axs[i,j].plot(topic_trend_dates, sorted_topic_trends[:,k+1], ls='--', color=color_list[color_counter])
				axs[i,j].tick_params(labelsize=18)
				axs[i,j].tick_params(axis='x', rotation=25)
				axs[i,j].set_title("Topic {}".format(k), fontsize=20)
				k+=1
				color_counter+=1
		plt.tight_layout(pad=3.0) 
		plt.savefig("results/weekly_trends.png")
		plt.close()
		plt.clf()

		weekly_period_file = open("results/weekly_periods.txt", "w")
		fig, axs = plt.subplots(3, 3, figsize=(24,15))
		k = 0
		color_counter = 0
		for i in range(3):
			for j in range(3):
				while k not in topics_sorted_by_coherence:
					k+=1
				freqs, power_spectral_density = signal.periodogram(sorted_topic_trends[:,k+1])
				power_spectrum_idx = np.argsort(power_spectral_density)
				weekly_period_file.write("Topic {} Top Five Periods (weeks): ".format(k) + str(1/freqs[power_spectrum_idx][::-1][:5]) + "\n")
				axs[i,j].semilogy(freqs, np.sqrt(power_spectral_density), color=color_list[color_counter])
				axs[i,j].set_ylim([1e0, 1e3])
				axs[i,j].tick_params(labelsize=18)
				axs[i,j].set_title("Topic {}".format(k), fontsize=20)
				k+=1
				color_counter+=1
		plt.tight_layout(pad=3.0) 
		plt.savefig("results/weekly_periodogram.png")
		plt.close()
		plt.clf()
		weekly_period_file.close()


		weekly_slope_file = open("results/weekly_slopes.txt", "w")
		fig, axs = plt.subplots(3, 3, figsize=(24,15))
		time_weeks = ((sorted_topic_trends[:,0] - sorted_topic_trends[0,0])/(7*24*3600)).reshape(-1,1)
		k = 0
		color_counter = 0
		for i in range(3):
			for j in range(3):
				while k not in topics_sorted_by_coherence:
					k+=1
				axs[i,j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
				reg = TheilSenRegressor(n_jobs=-1).fit(time_weeks, sorted_topic_trends[:,k+1])
				weekly_slope_file.write("Topic {} Slope: {:.4e}\n".format(k, reg.coef_[0]))
				axs[i,j].plot(topic_trend_dates, reg.intercept_ + reg.coef_[0]*time_weeks.flatten(), c='r',label="Slope: {:.2e}".format(reg.coef_[0]))
				axs[i,j].plot(topic_trend_dates, sorted_topic_trends[:,k+1], ls='-', color=color_list[color_counter])
				axs[i,j].tick_params(labelsize=18)
				axs[i,j].tick_params(axis='x', rotation=25)
				axs[i,j].set_title("Topic {}".format(k), fontsize=20)
				axs[i,j].legend(fontsize=16)
				k+=1
				color_counter+=1
		plt.tight_layout(pad=3.0) 
		plt.savefig("results/weekly_theilsen.png")
		plt.close()
		plt.clf()
		weekly_slope_file.close()




	if args.daily_series:
		'''
		For each day in the data, collect comments, tokenize them, and form
		BoW vectors based on pre-defined dictionary. Then, calculate how many times
		each topic is the predominant topic of a comment for each day. Plot this as
		a time series, plot the periodogram (reporting top five periods), and plot
		the Theil-Sen slope estimate.
		'''

		full_dataframe = pd.DataFrame()
		for data_file in os.listdir('weekly_data'):
			loaded_comments = pd.read_csv('weekly_data/' + data_file)
			full_dataframe = pd.concat([full_dataframe,loaded_comments], axis=0)

		# create dictionary of tokenized comment lists for each day
		num_days = (topic_trend_dates[-1] - topic_trend_dates[0] + dt.timedelta(days=7)).days
		start_datetime = topic_trend_dates[0]
		comments_list_daily = dict()
		for i in range(num_days):
			this_start = (start_datetime + dt.timedelta(days=i)).timestamp()
			this_end = (start_datetime + dt.timedelta(days=1+i)).timestamp()
			loaded_comments = list(full_dataframe[full_dataframe['created_utc'].between(this_start,this_end)]['body'])
			tokenized_loaded_comments, _, _ = tokenize_lda(data_cleaning(loaded_comments), bigram_model=bigram_model, trigram_model=trigram_model)
			bow_loaded_comments = [dictionary.doc2bow(comment) for comment in tokenized_loaded_comments]
			comments_list_daily[this_start] = bow_loaded_comments

		# create dictionary tracking how frequently each topic is present per day
		topic_multiplicity_daily = dict()
		for key in comments_list_daily.keys():
			topic_multiplicity_daily[key] = np.zeros(args.n_topics)
			this_day_comments = comments_list_daily[key]
			for comment in this_day_comments:
				topic_dist = lda.get_document_topics(comment,per_word_topics=False)
				top_topic_idx = np.argmax(np.array(topic_dist)[:,1])
				top_topic = topic_dist[top_topic_idx][0]
				topic_multiplicity_daily[key][top_topic] += 1

		# transform this dictionary into a sorted array
		daily_topic_trends = []
		for key in topic_multiplicity_daily.keys():
			daily_topic_trends.append([key]+list(topic_multiplicity_daily[key]))
		daily_topic_trends = np.array(daily_topic_trends)
		daily_sorted_idx = np.argsort(daily_topic_trends[:,0])
		sorted_daily_topic_trends = daily_topic_trends[daily_sorted_idx]
		daily_topic_trend_dates = [dt.datetime.fromtimestamp(stamp) for stamp in sorted_daily_topic_trends[:,0]]

		color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',
					  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

		fig, axs = plt.subplots(3, 3, figsize=(24,15))
		k = 0
		color_counter = 0
		for i in range(3):
			for j in range(3):
				while k not in topics_sorted_by_coherence:
					k+=1
				axs[i,j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
				axs[i,j].plot(daily_topic_trend_dates, sorted_daily_topic_trends[:,k+1], ls='-', color=color_list[color_counter])
				axs[i,j].tick_params(labelsize=18)
				axs[i,j].tick_params(axis='x', rotation=25)
				axs[i,j].set_title("Topic {}".format(k), fontsize=20)
				k+=1
				color_counter+=1
		plt.tight_layout(pad=3.0) 
		plt.savefig("results/daily_trends.png")
		plt.close()
		plt.clf()

		daily_period_file = open("results/daily_periods.txt", "w")
		fig, axs = plt.subplots(3, 3, figsize=(24,15))
		k = 0
		color_counter = 0
		for i in range(3):
			for j in range(3):
				while k not in topics_sorted_by_coherence:
					k+=1
				freqs, power_spectral_density = signal.periodogram(sorted_daily_topic_trends[:,k+1])
				power_spectrum_idx = np.argsort(power_spectral_density)
				daily_period_file.write("Topic {} Top Five Periods (days): ".format(k) + str(1/freqs[power_spectrum_idx][::-1][:5]) + "\n")
				axs[i,j].semilogy(freqs, np.sqrt(power_spectral_density), color=color_list[color_counter])
				axs[i,j].set_ylim([1e-1, 1e3])
				axs[i,j].tick_params(labelsize=18)
				axs[i,j].set_title("Topic {}".format(k), fontsize=20)
				k+=1
				color_counter+=1
		plt.tight_layout(pad=3.0) 
		plt.savefig("results/daily_periodogram.png")
		plt.close()
		plt.clf()
		daily_period_file.close()

		daily_slope_file = open("results/daily_slopes.txt", "w")
		fig, axs = plt.subplots(3, 3, figsize=(24,15))
		time_days = ((sorted_daily_topic_trends[:,0] - sorted_daily_topic_trends[0,0])/(24*3600)).reshape(-1,1)
		k = 0
		color_counter = 0
		for i in range(3):
			for j in range(3):
				while k not in topics_sorted_by_coherence:
					k+=1
				axs[i,j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
				reg = TheilSenRegressor(n_jobs=-1).fit(time_days, sorted_daily_topic_trends[:,k+1])
				daily_slope_file.write("Topic {} Slope: {:.4e}\n".format(k, reg.coef_[0]))
				axs[i,j].plot(daily_topic_trend_dates, reg.intercept_ + reg.coef_[0]*time_days.flatten(), c='r',label="Slope: {:.2e}".format(reg.coef_[0]))
				axs[i,j].plot(daily_topic_trend_dates, sorted_daily_topic_trends[:,k+1], ls='-', color=color_list[color_counter])
				axs[i,j].tick_params(labelsize=18)
				axs[i,j].tick_params(axis='x', rotation=25)
				axs[i,j].set_title("Topic {}".format(k), fontsize=20)
				axs[i,j].legend(fontsize=16)
				k+=1
				color_counter+=1
		plt.tight_layout(pad=3.0)
		plt.savefig("results/daily_theilsen.png")
		plt.close()
		plt.clf()
		daily_slope_file.close()