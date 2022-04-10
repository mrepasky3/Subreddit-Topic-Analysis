import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates
import statsmodels.api as sm
from statsmodels.tsa.stattools import pacf
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n_topics', type=int, default=15)
parser.add_argument('--topic_model', type=str, default='lda', choices=['lda', 'nmf', 'transformer'])
parser.add_argument('--stackplot', action="store_true")
parser.add_argument('--gridplot', action="store_true")
parser.add_argument('--partial_corr', action="store_true")
parser.add_argument('--partial_corr_full', action="store_true")
parser.add_argument('--full_corr_target', type=int)
parser.add_argument('--VAR', action="store_true")
parser.set_defaults(feature=False)


def generate_stackplot(dates, topic_freqs, savename):
	'''
	Create a stackplot representing frequency of each
	topic throughout the dates provided in dates.

	Parameters
	----------
	dates : list of datetime
		dates corresponding to each freqeuncy metric
	topic_freqs : (N, n_topics+1) numpy array
		first column is timestamp, remaining columns
		represent frequency of each topic on each date
	savename : str
		name for saving the image in the results folder
	'''
	fig = plt.figure(figsize=(25,10))
	plt.axvspan(dates[0], dates[-1], color='black')
	plt.stackplot(dates, topic_freqs[:,1:].T, labels=["Topic "+str(k) for k in range(topic_freqs.shape[1]-1)], zorder=3)

	plt.ylabel('Topic Proportion', fontsize=26)

	plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
	plt.tick_params(labelsize=22)
	plt.tick_params(axis='x', rotation=22)
	plt.xlabel('Date', fontsize=26)

	plt.margins(0,0)
	plt.legend(fontsize=16)
	plt.savefig("results/{}_stackplot.png".format(savename))
	plt.close()
	plt.clf()


def generate_gridplot(dates, topic_freqs, savename):
	'''
	Create a grid of plots representing frequency of each
	topic throughout the dates provided in dates.

	Parameters
	----------
	dates : list of datetime
		dates corresponding to each freqeuncy metric
	topic_freqs : (N, n_topics+1) numpy array
		first column is timestamp, remaining columns
		represent frequency of each topic on each date
	savename : str
		name for saving the image in the results folder
	'''
	color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
			  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
			  'chartreuse', 'dodgerblue', 'indigo', 'blue', 'teal']

	fig, axs = plt.subplots(5, 3, sharey=False, figsize=(24,25))
	k = 0
	for i in range(5):
		for j in range(3):
			axs[i,j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
			axs[i,j].plot(dates, topic_freqs[:,k+1], ls='-', color=color_list[k])
			axs[i,j].tick_params(labelsize=18)
			axs[i,j].tick_params(axis='x', rotation=25)
			axs[i,j].set_title("Topic {}".format(k), fontsize=20)
			axs[i,j].set_ylim(topic_freqs[:,k+1].mean()-3*topic_freqs[:,k+1].std(), topic_freqs[:,k+1].mean()+5*topic_freqs[:,k+1].std())
			k+=1
	fig.text(0.0, 0.5, 'Topic Frequency', va='center', rotation='vertical',fontsize=25)
	plt.tight_layout(pad=3.0) 
	plt.savefig("results/{}_gridplot.png".format(savename))
	plt.close()
	plt.clf()


def partial_correlation(feature_series, target_series, lags=365):
	'''
	Calculate the partial correlation between the target
	series and the various lags of the feature series. "Partial"
	refers to the fact that the calculation of correlation controls
	for the effect of more recent lags.

	Parameters
	----------
	feature_series : pandas Series
		time series used for prediction
	target_series : pandas Series
		time series to be predicted
	lags : int
		largest lag to check for partial correlation

	Returns
	-------
	pcf : list of float
		partial correlation vaules for each lag in [1,lags)
	pvals : list of float
		p-value for the correlation calculations
	'''
	pcf = []
	pvals = []
	data_df = pd.DataFrame([feature_series, target_series]).T
	for i in tqdm(range(1,lags)):
		this_dataframe = data_df.copy()
		for j in range(i):
			this_dataframe = pd.concat([feature_series.copy().shift(j+1), this_dataframe], axis=1)
		this_dataframe = this_dataframe.dropna().reset_index(drop=True)
		this_dataframe.columns = [str(feature_series.name) + " Lag {}".format(lag) for lag in np.arange(0,i+1)[::-1]] + [target_series.name]
		
		X = this_dataframe.iloc[:,0:-1]
		X = sm.add_constant(X)
		
		results=sm.OLS(this_dataframe.iloc[:,-1],X).fit()

		pcf.append(results.params[this_dataframe.columns[0]])
		pvals.append(results.pvalues[this_dataframe.columns[0]])
	return pcf, pvals


def partial_correlation_allfields(target_index, full_data, lags=50, write=False, topic_model='lda'):
	'''
	Calculate the partial correlation between the target
	series and the various lags of all features. "Partial"
	refers to the fact that the calculation of correlation controls
	for the effect of more recent lags of all fields.

	Parameters
	----------
	target_index : int
		index of time series to be predicted
	full_data : pandas DataFrame
		dataframe containing all time series including
		the target (target_idx refers to this frame)
	lags : int
		largest lag to check for partial correlation
	write : bool
		indicate whether or not to save the results as
		they are generated
	topic_model : str
		model name for file saving purposes

	Returns
	-------
	pcf : list of float
		partial correlation vaules for each lag in [1,lags)
	pvals : list of float
		p-value for the correlation calculations
	'''
	pcf = []
	pvals = []
	if write:
		pcf_file = open("results/{}_topic{}_fulldata_partial_corr_table.csv".format(topic_model, target_index), 'w')
		pcf_file.write("lag," + ",".join(["topic {}".format(k) for k in range(full_data.shape[1])]) + "\n")
		pval_file = open("results/{}_topic{}_fulldata_pval_table.csv".format(topic_model, target_index), 'w')
		pval_file.write("lag," + ",".join(["topic {}".format(k) for k in range(full_data.shape[1])]) + "\n")
	target_df = pd.DataFrame(full_data.iloc[:,target_index])
	for i in tqdm(range(1,lags)):
		this_dataframe = target_df.copy()
		for j in range(i):
			this_dataframe = pd.concat([full_data.copy().shift(j+1), this_dataframe], axis=1)
		this_dataframe = this_dataframe.dropna().reset_index(drop=True)
		column_list = []
		for lag in np.arange(1,i+1)[::-1]:
			column_list += list(np.char.add(np.array(full_data.columns, dtype=str),
								   np.array([' Lag {}'.format(lag)] * full_data.shape[1])))
		this_dataframe.columns = column_list + [str(full_data.iloc[:,target_index].name)]
		
		X = this_dataframe.iloc[:,0:-1]
		X = sm.add_constant(X)
		
		results=sm.OLS(this_dataframe.iloc[:,-1],X).fit()

		pcf.append(list(results.params[1:full_data.shape[1]+1]))
		pvals.append(list(results.pvalues[1:full_data.shape[1]+1]))
		if write:
			pcf_file.write("{},".format(i) + ",".join(np.array(pcf[-1], dtype=str)) + "\n")
			pval_file.write("{},".format(i) + ",".join(np.array(pvals[-1], dtype=str)) + "\n")
	if write:
		pcf_file.close()
		pval_file.close()
	return pcf, pvals


def vector_autoregression(topic_freqs, target_index, pval_table, signif=0.05, train_split=250, mode='all'):
	'''
	Using partial correlation results, construct a vector
	autoregression model that uses the coefficients with
	lowest p-value.

	Parameters
	----------
	topic_freqs : pandas DataFrame
		time series of each of the topics
	target_index : int
		index of the prediction variable
	pval_table : pandas DataFrame
		p values for all lags of the data
	signif : float
		degree of significance to keep
		variables for use in the model
	train_split : int
		index of the train-test split
	mode : 'all', 'auto', or 'nonauto'
		use all data if 'all', use
		only the same time series if 'auto',
		and use everything except the same
		time series if 'nonauto'

	Returns
	-------
	results : statsmodels RegressionResults
		result after fitting linear model to training data
	X_train : pandas DataFrame
		lagged data used as features for the regression training
	X_test : pandas DataFrame
		lagged feature data for testing
	y_train : pandas Series
		training portion of target topic frequency
	y_test : pandas Series
		testing portion of target topic frequency
	y_pred : pandas Series
		test prediction of the trained model
	mae_test : float
		mean absolute error of the test prediction
	'''

	target_pvals = []
	for i in range(pval_table.shape[1]):
		if int(pval_table.columns[i].split()[-1]) == target_index:
			target_pvals.append(pval_table.iloc[:,i])
	target_pvals = pd.DataFrame(target_pvals).T

	
	target_lags = []
	if mode == 'all':
		for i in range(target_pvals.shape[1]):
			target_lags.append(np.array(target_pvals.index[target_pvals.iloc[:,i]<signif]) + 1)
	
	elif mode == 'auto':
		for i in range(target_pvals.shape[1]):
			if i != target_index:
				target_lags.append(np.array([]))
			else:
				target_lags.append(np.array(target_pvals.index[target_pvals.iloc[:,i]<signif]) + 1)
				
	elif mode == 'nonauto':
		for i in range(target_pvals.shape[1]):
			if i == target_index:
				target_lags.append(np.array([]))
			else:
				target_lags.append(np.array(target_pvals.index[target_pvals.iloc[:,i]<signif]) + 1)
		
	
	regression_df = topic_freqs.iloc[:,target_index].copy()
	for i in range(len(target_lags)):
		if len(target_lags[i]) != 0:
			for lag in target_lags[i]:
				regression_df = pd.concat([topic_freqs.iloc[:,i].copy().shift(lag), regression_df], axis=1)
	regression_df = regression_df.dropna().reset_index(drop=True)
	
	
	X = regression_df.iloc[:,:-1]
	X = sm.add_constant(X)
	X_train, X_test = X.iloc[:train_split], X.iloc[train_split:]
	model = sm.OLS(regression_df.iloc[:train_split,-1], X_train)
	results = model.fit()
	
	y_pred = results.predict(X_test)
	mae_test = np.mean(np.abs(y_pred - regression_df.iloc[train_split:,-1]))
	
	return results, X_train, X_test, regression_df.iloc[:train_split,-1], regression_df.iloc[train_split:,-1], y_pred, mae_test




if __name__ == '__main__':
	
	args = parser.parse_args()

	if args.topic_model == 'lda':
		daily_topic_freqs = np.load("results/lda_daily_trends.npy")
		daily_topic_df = pd.DataFrame(daily_topic_freqs[:,1:])
		dates = [dt.datetime.fromtimestamp(stamp) for stamp in daily_topic_freqs[:,0]]

		pcf_table = pd.read_csv("results/lda_partial_corr_table.csv")
		pval_table = pd.read_csv("results/lda_pval_table.csv")


	if args.stackplot:
		generate_stackplot(dates, daily_topic_freqs, savename=args.topic_model + "_daily")


	if args.gridplot:
		generate_gridplot(dates, daily_topic_freqs, savename=args.topic_model + "_daily")


	if args.partial_corr:
		n_topics = daily_topic_df.shape[1]
		partial_corr_table = np.zeros((200-1,n_topics**2))
		pval_table = np.zeros((200-1,n_topics**2))
		titles = []
		
		iteration = 0
		for k1 in range(n_topics):
			for k2 in range(n_topics):
				feature = daily_topic_df.iloc[:,k1]
				target = daily_topic_df.iloc[:,k2]
				pcf, pvals = partial_correlation(feature, target)
				titles.append("Feature {} Target {}".format(k1, k2))
				partial_corr_table[:,iteration] = pcf
				pval_table[:,iteration] = pvals
				iteration+=1

		pd.DataFrame(partial_corr_table,columns=titles).to_csv("results/"+args.topic_model+"_partial_corr_table.csv", index=False)
		pd.DataFrame(pval_table,columns=titles).to_csv("results/"+args.topic_model+"_pval_table.csv", index=False)


	if args.partial_corr_full:
		target_idx = args.full_corr_target
		partial_correlation_allfields(target_idx, daily_topic_df, write=True, topic_model=args.topic_model)


	if args.VAR:
		os.mkdir("results/{}_VAR/".format(args.topic_model))
		report_file = open("results/{}_VAR/results.txt".format(args.topic_model))
		for i in range(pval_table.shape[1]):
			res, X_train, X_test, y_train, y_test, y_pred, mae_test = vector_autoregression(daily_topic_df, i, pval_table.iloc[:100,:],
				signif=0.05, train_split=200, mode='all')

			report_file.write("Target {}\nF p-value {:.2e}\nR^2 {:.2f}\n".format(i, res.f_pvalue, res.rsquared))
			report_file.write("Test MAE {:.2e}\nNum Params {}\nNum_Significant {}".format(mae_test, len(res.params), (res.pvalues < 0.05).sum()))

			preds = res.get_prediction(pd.concat([X_train,X_test]))

			fig = plt.figure(figsize=(15,5))

			plt.plot(y_train, color='k')
			plt.plot(y_test, color='k')
			plt.axvline(200, color='red')

			plt.plot(preds.summary_frame()['mean'], ls='--', color='r')
			plt.fill_between(range(len(preds.summary_frame()['mean'])),
							 preds.summary_frame()['mean_ci_lower'], preds.summary_frame()['mean_ci_upper'], color='r', alpha=.1)

			plt.show()