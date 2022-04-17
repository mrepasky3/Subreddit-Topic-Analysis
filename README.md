# Understanding Community Trends in Online Mental Health Forums (Subreddit-Topic-Analysis)

This code corresponds to the [CS 6471 Computational Social Science](https://www.cc.gatech.edu/classes/AY2022/cs6471_spring/) at Georgia Tech semester project. In this project, we conduct a Reddit-based case study on
college students which suffer from mental-health-related concerns and/or particpate in online discussion regarding mental health. To do so, we gather Reddit comment data
from December 2019 - December 2021 using the [Pushshift Reddit API](https://github.com/pushshift/api). We select comments from users posting in both mental health subreddits
(such as [r/Anxiety](https://www.reddit.com/r/Anxiety) and [r/Depression](https://www.reddit.com/r/Depression)) and in college-related subreddits (such as 
[r/college](https://www.reddit.com/r/college) and [r/gatech](https://www.reddit.com/r/gatech)). A series of topic models (LDA, NMF, and a CWE-based model)
are fit to the data, followed by a time series partial correlation anaylsis and the construction of a set of vector autoregression models (one VAR model
per topic per topic model).

The files have the following roles:  
1. Data Gathering: `extract_weekly_data_*_.py` files use the gathered list of subreddits and the pushshift API to pull comment data from Reddit and deposit them into files spanning one week each of data.
2. Topic Modeling: `*_driver.py` files use utility functions in `utils.py` to clean, tokenize, and otherwise preprocess these data before fitting and visualizing LDA, NMF, and contextual topic models.
3. Time Series Analysis: the `generate_time_series_*` methods in `utils.py` are used to generate time series arrays for each fitted model, and `time_series_analysis.py` visualizes these data, plots them as stack plots and line plots, performs partial correlation analysis, and fits VAR models.

Much of the LDA/NMF topic modeling, pre-processing, and evaluation is adapted to our problem setting using the [Evaluate Topic Models: Latent Dirichlet Allocation (LDA)](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0) Towards Data Science article by Shashank Kapadia. The CTM is implemented using the [Contextualized Topic Models](https://github.com/MilaNLProc/contextualized-topic-models) Python module.
