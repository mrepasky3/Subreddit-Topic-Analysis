'''
Referenced from
https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
'''
import re
import os
import numpy as np
import pandas as pd
import datetime as dt
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from nltk.corpus import stopwords




def data_cleaning(comments_list):
	'''
	Remove links, control sequences, emojis, and likely bots

	Parameters
	----------
	comments_list : list of str
		list of raw data strings representing
		comments pulled from pushshift Reddit API

	Returns
	-------
	cleaned_comments : list of str
		list of strings representing cleaned
		comments which removed bots and certain
		characters and punctuation
	'''

	cleaned_comments = []
	for i in range(len(comments_list)):
		# remove brackets
		semi_clean = np.array(re.sub('\]',' ',re.sub('\[','',comments_list[i])).split())
		# remove links
		mask = [('http' not in word) and ('Http' not in word) and ('www.' not in word) and ('Www.' not in word) and ('.com' not in word) for word in semi_clean]
		semi_clean = " ".join(list(semi_clean[mask]))
		# remove control sequences
		semi_clean = re.sub('&gt;','',semi_clean)
		semi_clean = re.sub('&amp;','',semi_clean)
		semi_clean = re.sub('\n','',semi_clean)
		semi_clean = re.sub("\'","",semi_clean)
		semi_clean = re.sub("\*","",semi_clean)
		semi_clean = semi_clean.replace("\\","")
		semi_clean = re.sub("’","",semi_clean)
		# remove emojis
		# from https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
		emoji_pattern = re.compile("["
							   u"\U0001F600-\U0001F64F"  # emoticons
							   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
							   u"\U0001F680-\U0001F6FF"  # transport & map symbols
							   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
							   u"\U00002702-\U000027B0"
							   u"\U000024C2-\U0001F251"
							   u"\U0001F973" # part hat emoji
							   u"\U0001F97A" # sad eyes emoji
							   u"\U0001F970" # smiling face with 3 hearts
							   "]+", flags=re.UNICODE)
		semi_clean = re.sub(emoji_pattern,r'', semi_clean)
		# remove punctuation and symbols
		clean = re.sub('[./&%$#()!?;:]', '', semi_clean).lower()
		# remove empty comments and comments posted by bots
		if (len(clean) > 0) and ('ɴᴏᴡ ᴘʟᴀʏɪɴɢ' not in clean) and ('bleep' not in clean) and ('beep' not in clean):
			if ('[view link]' not in clean) and ('i am a bot' not in clean) and ('dmca removal request for' not in clean):
				cleaned_comments.append(clean)

	return cleaned_comments


def tokenize_lda(cleaned_comments, bigram_model=None, trigram_model=None):
	'''
	Tokenize each cleaned comment, remove stopwords, and
	create bigram/trigrams

	Parameters
	----------
	cleaned_comments : list of str 
		list containing pre-processed, cleaned comments
	bigram_model : gensim Phrases
		pre-trained model for 2-grams, a new model is
		trained if this field is None
	trigram_model : gensim Phrases
		pre-trained model for 3-grams, a new model is
		trained if this field is None

	Returns
	-------
	comments_words : list of list of str
		each list corresponds to a tokenized comments,
		where the tokens are cleaned of stopwords and
		include 2-grams and 3-grams.
	bigram_model : gensim Phrases
		trained model for 2-grams, or same as input
		parameter if not None
	trigram_model : gensim Phrases
		trained model for 3-grams, or same as input
		parameter if not None
	'''

	stop_words = stopwords.words('english')
	stop_words.extend(['im','ive','dont','get','youre','would','thats',
					   'really','one','also','something','even','thing','things','must',
					   'cant','much','could','way','lot','got','get','go','like','th'])

	# Convert each comment to a list of words, removing any missed punctuation
	word_posts = []
	for post in cleaned_comments:
		word_posts.append(gensim.utils.simple_preprocess(post, deacc=True))
	comments_words = word_posts

	# Remove words which are very common
	cleaned_posts = []
	for post in comments_words:
		cleaned_post = []
		for word in post:
			if word not in stop_words:
				cleaned_post.append(word)
		cleaned_posts.append(cleaned_post)
	comments_words = cleaned_posts

	# Create bigrams - two words which are commonly used together
	if bigram_model == None:
		bigram_model = gensim.models.Phrases(comments_words, min_count=15, threshold=200)
	bigrammized_posts = []
	for post in comments_words:
		bigrammized_posts.append(bigram_model[post])
	comments_words = bigrammized_posts

	# Create trigrams - three words which are commonly used together
	if trigram_model == None:
		trigram_model = gensim.models.Phrases(bigram_model[comments_words], threshold=200)
	trigrammized_posts = []
	for post in comments_words:
		trigrammized_posts.append(trigram_model[bigram_model[post]])
	comments_words = trigrammized_posts

	return comments_words, bigram_model, trigram_model


def create_dictionary(comments_words):
	'''
	Generate gensim dictionary for tokenized comment corpus

	Parameters
	----------
	comments_words : list of list of str
		each list corresponds to a tokenized comments,
		where the tokens are cleaned of stopwords and
		include 2-grams and 3-grams.

	Returns
	-------
	comments_bow : list of list of (int, int)
		list of bow representations of
		comments in corpus, which are lists
		of token-multiplicity tuples
	dictionary : gensim Dictionary
		maps token indices to tokens
	'''

	dictionary = corpora.Dictionary(comments_words)
	comments_bow = [dictionary.doc2bow(comment) for comment in comments_words]
	
	return comments_bow, dictionary


def generate_time_series_lda(lda, bigram_model, trigram_model, dictionary, save=False):
	'''
	Given fit LDA, n-gram models, and dictionary, gather all the
	data, convert to BoW representations of each comment made in
	each day, assign each comment to the most-probable topic, record
	portion of each days comments belonging to each topic, return
	and save this array sorted in time.

	Parameters
	----------
	lda : gensim LdaMulticore
		trained latent Dirichlet allocation model
	bigram_model : gensim Phrases
		fit model which converts word pairs in lists of tokens to bigrams
	trigram_model : gensim Phrases
		fit model which converts bigrams in lists of tokens to trigrams
	dictionary : gensim Dictionary
		dictionary used for fitting LDA model
	save : bool
		indicate whether or not the daily trend array should
		be saved as a numpy file

	Returns
	-------
	daily_topic_trend_dates : list of datetime
		sorted list of datetime objects corresponding to
		the first column of timestamps in the topic frequency array
	sorted_daily_topic_trends : numpy array
		temporally-sorted (num_days x num_topics+1) array which stores
		the unix timestamp as the first column and the portion of comments
		belonging to each topic for each corresponding day as each
		of the remaining columns
	'''
	full_dataframe = pd.DataFrame()
	for data_file in os.listdir('weekly_data'):
	    loaded_comments = pd.read_csv('weekly_data/' + data_file)
	    full_dataframe = pd.concat([full_dataframe,loaded_comments], axis=0)

	# determine the beginning and end time of the data
	start_dates = []
	for data_file in os.listdir('weekly_data'):
	    start = data_file.split('-')[0][-10:]
	    start_time = int(dt.datetime.strptime(start, '%d_%m_%Y').timestamp())
	    start_dates.append(start_time)
	start_dates = np.array(start_dates)
	sorted_idx = np.argsort(start_dates)
	sorted_start_dates = start_dates[sorted_idx]
	topic_trend_dates = [dt.datetime.fromtimestamp(stamp) for stamp in sorted_start_dates]

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
	    topic_multiplicity_daily[key] = np.zeros(15)
	    this_day_comments = comments_list_daily[key]
	    for comment in this_day_comments:
	        topic_dist = lda.get_document_topics(comment,per_word_topics=False)
	        top_topic_idx = np.argmax(np.array(topic_dist)[:,1])
	        top_topic = topic_dist[top_topic_idx][0]
	        topic_multiplicity_daily[key][top_topic] += 1
	    if len(this_day_comments) != 0 :
	    	topic_multiplicity_daily[key] /= len(this_day_comments)

	# transform this dictionary into a sorted array
	daily_topic_trends = []
	for key in topic_multiplicity_daily.keys():
	    daily_topic_trends.append([key]+list(topic_multiplicity_daily[key]))
	daily_topic_trends = np.array(daily_topic_trends)
	daily_sorted_idx = np.argsort(daily_topic_trends[:,0])
	sorted_daily_topic_trends = daily_topic_trends[daily_sorted_idx]
	daily_topic_trend_dates = [dt.datetime.fromtimestamp(stamp) for stamp in sorted_daily_topic_trends[:,0]]

	if save:
		np.save("results/lda_daily_trends.npy",sorted_daily_topic_trends)

	return daily_topic_trend_dates, sorted_daily_topic_trends