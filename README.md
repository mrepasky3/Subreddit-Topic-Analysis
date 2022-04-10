# Understanding Community Trends in Online Mental Health Forums (Subreddit-Topic-Analysis)

This code corresponds to the [CS 6471 Computational Social Science](https://www.cc.gatech.edu/classes/AY2022/cs6471_spring/) at Georgia Tech semester project. In this project, we conduct a Reddit-based case study on
college students which suffer from mental-health-related concerns and/or particpate in online discussion regarding mental health. To do so, we gather Reddit comment data
from December 2019 - December 2021 using the [Pushshift Reddit API](https://github.com/pushshift/api). We select comments from users posting in both mental health subreddits
(such as [r/Anxiety](https://www.reddit.com/r/Anxiety) and [r/Depression](https://www.reddit.com/r/Depression)) and in college-related subreddits (such as 
[r/college](https://www.reddit.com/r/college) and [r/gatech](https://www.reddit.com/r/gatech)). A series of topic models (LDA, NMF, and a CWE-based clustering)
are conducted, followed by an time series partial correlation anaylsis and the construction of a set of vector autoregression models (one VAR model
per topic per topic model).

The files fulfill the following roles:  
1. Data Gathering: `extract_weekly_data_*_.py` files use the gathered list of subreddits and the pushshift API to pull comment data from Reddit and deposit them into files spanning one week each of data.
2. Topic Modeling: `driver.py` uses utility functions in `utils.py` to clean and tokenize these data before fitting and visualizing an LDA model, followed by preliminary visualization of the time series.
3. Time Series Analysis: the `generate_time_series_lda` method in `utils.py` is used to generate a time series array for the LDA model, and `time_series_analysis.py` visualizes these data, plots them as stack plots and line plots, performs partial correlation analysis, and fits VAR models).

Much of the LDA topic modeling, pre-processing, and evaluation is adapted to our problem setting using the [Evaluate Topic Models: Latent Dirichlet Allocation (LDA)](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0) Towards Data Science article by Shashank Kapadia.
