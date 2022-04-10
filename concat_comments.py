import os
import argparse
import pandas as pd
import numpy as np
import glob
import json



parser = argparse.ArgumentParser()
parser.add_argument('--n_topics', type=int, default=15)
parser.add_argument('--weekly_series', action="store_true")
parser.add_argument('--daily_series', action="store_true")
parser.add_argument('--load', action="store_true")
parser.set_defaults(feature=False)




if __name__ == '__main__':
	files = os.path.join('weekly_data/', '*.csv')

	files = glob.glob(files)

	df = pd.concat(map(pd.read_csv, files), ignore_index=True)

	df.to_csv('comments.csv', index=False)
	# comment_list = df['body'].tolist()
	# comments_small = comment_list[:100]
	# # with open('comments.txt', 'w') as f:
	# # 	for comment in comment_list:
	# # 		f.write("%s\n" % comment)
	# file = open('comments_small.csv', 'w+', newline ='')
	
	# #print(comments_small[:10])

	# with open("comments_small.txt", "w") as f:
	# 	for i in range(100):
	# 		f.write("%s\n" % comment_list[i])
	
	print(df.info())
