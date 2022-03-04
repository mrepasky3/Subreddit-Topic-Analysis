from pmaw import PushshiftAPI
import os
import pandas as pd
import numpy as np
import datetime as dt

mental_health_subreddits = ['mentalhealth', 'Lonely', 'anxiety', 'raisedbynarcissists', 'narcissisticabuse', 'depressed',
						'depression', 'depression_help', 'disorder', 'addiction', 'adhd',
						'Anger', 'alcoholism', 'cripplingalcoholism', 'autism', 'aspergers',
						'ARFID', 'BingeEatingDisorder','bipolar', 'BipolarReddit',
						'BipolarSOs', 'BodyAcceptance', 'BPD', 'Bulimia', 'domesticviolence',
						'downsyndrome','BodyDysmorphia', 'Dysmorphicdisorder', 'Eatingdisorders',
						'emetophobia', 'EOOD', 'ForeverAlone', 'GFD', 'mentalillness', 'MMFB',
						'neurodiversity', 'OCD', 'OCPD', 'Phobia','PTSD', 'rapecounseling', 'rape', 'sad',
						'schizophrenia', 'schizoaffective', 'selfharm', 'selfhelp','socialanxiety',
						'socialskills', 'selfharm', 'survivorsofabuse', 'narcissisticparents', 'tourettes',
						'suicidewatch']

college_subreddits = ['rit','uchicago','ucf','berkeley','OSU','UIUC','msu','ryerson','Purdue','NCSU','ucla','iastate',
					  'USC','fsu','UTAustin','UBC','aggies','Cornell','VirginiaTech','UCSC','UCSD','McMaster','stanford',
					  'UCI','gatech','udub','uwaterloo','UofT','nyu','UCDavis','ASU','PennStateUniversity','mcgill','Pitt',
					  'rutgers','ufl','uofm','uwo','UMD','UniversityOfHouston','Concordia','uvic','uofmn','utdallas','UGA',
					  'yorku','UWMadison','WGU','UCSantaBarbara','UCalgary','uAlberta','geegees', 'college','collegerant',
					  'transferstudents','gradschool','gradadmissions','csmajors']


# save cached data, so querying later will be much faster (basically instant)
mental_health_cache = 'mental_health_cache'
if not os.path.exists(mental_health_cache):
	os.mkdir(mental_health_cache)

# save cached data, so querying later will be much faster (basically instant)
college_cache = 'college_cache'
if not os.path.exists(college_cache):
	os.mkdir(college_cache)

if not os.path.exists('weekly_data'):
	os.mkdir('weekly_data')


api = PushshiftAPI()


# January 3, 2021 at midnight UTC
overall_start_time = 1609632000


for week_index in range(0,18):
	start_time = overall_start_time + (604800*week_index)
	end_time = overall_start_time + (604800*(week_index+1))


	# query subreddits for comment data
	while True:
		# keep querying until successful
		try:
			mental_health_dfs = []
			for subreddit in mental_health_subreddits:
				print(f'Querying subreddit r/{subreddit}')
				comments = api.search_comments(subreddit=subreddit, limit=None, before=end_time, after=start_time,
											   safe_exit=True, cache_dir=mental_health_cache)
				mental_health_dfs.append(pd.DataFrame(comments))

			college_dfs = []
			for subreddit in college_subreddits:
				print(f'Querying subreddit r/{subreddit}')
				comments = api.search_comments(subreddit=subreddit, limit=None, before=end_time, after=start_time,
											   safe_exit=True, cache_dir='college_cache')
				college_dfs.append(pd.DataFrame(comments))
			break
		except:
			pass


	# Define the union of all users in these mental health subreddits
	mental_health_users = []
	for i in range(len(mental_health_dfs)):
		if len(mental_health_dfs[i]) > 0:
			mental_health_users += list(mental_health_dfs[i]['author'].unique())
	mental_health_users = pd.Series(mental_health_users).unique()


	# Define the union of all users in these college subreddits
	college_users = []
	for i in range(len(college_dfs)):
		if len(college_dfs[i]) > 0:
			college_users += list(college_dfs[i]['author'].unique())
	college_users = pd.Series(college_users).unique()


	# Define intersection of users in both groups of subredddits
	intersecting_users = list(set(list(mental_health_users)).intersection(list(college_users)))
	

	# Remove users that post identical comments since they are likely bots
	cleaned_intersecting_users = []
	for username in intersecting_users:
		if (username.lower() == 'automoderator') or (username.lower() == '[deleted]'):
			continue
		if ('bot' in username.lower()) or ('b0t' in username.lower()):
			continue
		
		comments = []
		num_comments = 0
		for i in range(len(college_dfs)):
			if len(college_dfs[i]) > 0:
				user_comments = college_dfs[i][college_dfs[i]['author'] == username]['body']
				if len(user_comments) != 0:
					for comment in user_comments:
						comments.append(comment)
					
		for i in range(len(mental_health_dfs)):
			if len(mental_health_dfs[i]) > 0:
				user_comments = mental_health_dfs[i][mental_health_dfs[i]['author'] == username]['body']
				if len(user_comments) != 0:
					for comment in user_comments:
						comments.append(comment)
						
		comment_set = set(comments)
		if (len(comment_set) != len(comments)):
			continue
		cleaned_intersecting_users.append(username)
	

	# compile all UTC times, subreddits, and comments into one dataframe for this week
	week_df = pd.DataFrame()
	for df in mental_health_dfs:
		if len(df) > 0:
			reduced_df = df[df['author'].isin(cleaned_intersecting_users)]
			reduced_df = pd.concat([reduced_df['created_utc'],reduced_df['subreddit'],reduced_df['body']],
										axis=1).reset_index(drop=True)
			week_df = pd.concat([week_df,reduced_df], axis=0).reset_index(drop=True)
	for df in college_dfs:
		if len(df) > 0:
			reduced_df = df[df['author'].isin(cleaned_intersecting_users)]
			reduced_df = pd.concat([reduced_df['created_utc'],reduced_df['subreddit'],reduced_df['body']],
										axis=1).reset_index(drop=True)
			week_df = pd.concat([week_df,reduced_df], axis=0).reset_index(drop=True)


	# save this weeks data into a csv file
	readable_start = dt.datetime.utcfromtimestamp(start_time).strftime('%d_%m_%Y')
	readable_end = dt.datetime.utcfromtimestamp(end_time).strftime('%d_%m_%Y')
	week_df.to_csv("weekly_data/mental_health_college_{}-{}.csv".format(readable_start,readable_end), index=False)