'''
Referenced from
https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
'''
import re
import numpy as np
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords




def data_cleaning(comments_list):
    '''
    Remove links, control sequences, emojis, and likely bots

    Parameters
    ----------
    comments_list : list of strings of raw comments

    Returns
    -------
    cleaned_comments : list of strings of cleaned comments
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
    cleaned_comments : list of strings of cleaned comments
    bigram_model : pre-trianed gensim Phrases model for 2-grams
    trigram_model : pre-trianed gensim Phrases model for 3-grams

    Returns
    -------
    comments_words : list of tokenized comments
    bigram_model : same as input, or fit to tokenized comments if input is None
    trigram_model : same as input, or fit to tokenized comments if input is None
    '''

    stop_words = stopwords.words('english')
    stop_words.extend(['im','ive','dont','get','youre','would','thats',
                       'really','one','also','something','even','thing','things','must',
                       'cant','much','could','way','lot','got','get','go','like','th'])

    # Convert each comment to a list of words, removing any missed punctuation
    word_posts = []
    for post in comments_words:
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
    comments_words : list of tokenized comments

    Returns
    -------
    comments_bow : list of bow representations of comments in corpus
    dictionary : gensim dictionary mapping indices to token
    '''

    dictionary = corpora.Dictionary(comments_words)
    comments_bow = [dictionary.doc2bow(comment) for comment in comments_words]
    
    return comments_bow, dictionary