from my_parser import *
from twiter_api import *
import pandas as pd
import numpy as np
from nltk.corpus import twitter_samples
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

database = pd.read_csv("nawl-analysis.csv")
my_en_stopwords = ['a', 'the', 'us', 'we', 'via', "'s", "'", "amp", 'i', '``', "n't", '...', "'", '`', '"']


def evaluate(words):
    dic = {
        'H': 0,
        'A': 0,
        'S': 0,
        'U': 0,
        'F': 0,
        'D': 0,
        'N': 0
    }
    for w in words:
        # print(database[database['word'] == w].index.values)
        if (database['word'] == w).any():
            index = database[database['word'] == w].index.values[0]
            category = database['category'][index]
            if category != 'U':
                dic[category] += 1
    v = list(dic.values())
    k = list(dic.keys())
    if max(v) != 0:
        return k[v.index(max(v))]
    else:
        return 'U'


def polish_tweets_analysis(filename):
    parser = Parser()
    tweets = pd.read_csv(filename)

    print("Read {} tweets".format(len(tweets)))
    results = {
        'H': 0,
        'A': 0,
        'S': 0,
        'U': 0,
        'D': 0,
        'F': 0,
        'N': 0
    }
    map = {
        'Hapiness': 'H',
        'Anger': 'A',
        'Sadness': 'S',
        'Undefined': 'U',
        'Disgust': 'D',
        'Fear': 'F',
        'Neutral': 'N'
    }
    for index, row in tweets.iterrows():
        words = parser.prepare_sentence(row['tweet'])
        result = evaluate(words)
        # print(result)
        # print(tweets[i].full_text)
        results[result] += 1  # + row['like_count'] + row['retweet_count']

    print(results)

    # Data to plot
    labels = map.keys()
    sizes = [results[v] for k, v in map.items()]
    # colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

    # Plot
    plt.pie(sizes, explode=explode, labels=labels,  # colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    tags = filename.split('.')[0].split('/')[-1]
    plt.title('{} - {} tweets'.format(tags, str(len(tweets))))
    plt.axis('equal')
    plt.show()


def cleanup_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    stop_words = stopwords.words('english')
    for word, tag in sentence:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        word = lemmatizer.lemmatize(word, pos).lower()
        if word not in string.punctuation and word not in stop_words and 'http' not in word and word not in my_en_stopwords and word != '"' and word != "'":
            lemmatized_sentence.append(word)
    return lemmatized_sentence


# def english_tweets_analysis():
#     categorized_tweets = ([(pos_tag(t), "p") for t in twitter_samples.tokenized("positive_tweets.json")] +
#                           [(pos_tag(t), "n") for t in twitter_samples.tokenized("negative_tweets.json")])
#     print(len(categorized_tweets))
#
#     lemmatized_tweets = []
#
#     for t, cat in categorized_tweets:
#         lemmatized_sentence = cleanup_sentence(t)
#
#         lemmatized_tweets.append((lemmatized_sentence, cat))
#         print(lemmatized_tweets)
#         break

def pl_freq_tweets_analysis(filename):
    tweets = pd.read_csv(filename)
    parser = Parser()

    full_list = []
    for index, row in tweets.iterrows():
        words = parser.prepare_sentence(row['tweet'])

        for w in words:
            full_list.append(w)
    freq_dist = nltk.FreqDist(full_list)
    data = [v for k, v in freq_dist.most_common(20)]
    labels = [k for k, v in freq_dist.most_common(20)]
    fig, ax = plt.subplots()
    width = 0.5  # the width of the bars
    ind = np.arange(len(data))  # the x locations for the groups
    ax.barh(ind, data, width, color="blue")
    ax.set_yticks(ind)
    ax.set_yticklabels(labels, minor=False)

    tags = filename.split('.')[0].split('/')[-1]
    plt.title('{} - {} tweets'.format(tags, str(len(tweets))))
    plt.xlabel('Wystąpienia')
    plt.ylabel('Słowo')
    plt.show()


def english_freq_tweets_analysis(filename):
    tweets = pd.read_csv(filename)
    print(len(tweets))

    lemmatized_tweets = []

    for index, row in tweets.iterrows():
        lemmatized_sentence = cleanup_sentence(pos_tag(word_tokenize(row['tweet'])))

        lemmatized_tweets.append(lemmatized_sentence)
    full_list = []

    for t in lemmatized_tweets:
        full_list += t

    freq_dist = nltk.FreqDist(full_list)
    data = [v for k, v in freq_dist.most_common(40)]
    labels = [k for k, v in freq_dist.most_common(40)]
    fig, ax = plt.subplots()
    width = 0.5  # the width of the bars
    ind = np.arange(len(data))  # the x locations for the groups
    ax.barh(ind, data, width, color="blue")
    ax.set_yticks(ind)
    ax.set_yticklabels(labels, minor=False)

    tags = filename.split('.')[0].split('/')[-1]
    plt.title('{} - {} tweets'.format(tags, str(len(tweets))))
    plt.xlabel('Wystąpienia')
    plt.ylabel('Słowo')
    plt.show()


def english_sentiment_tweets_analysis_vader(filename):
    tweets = pd.read_csv(filename)
    print(len(tweets))

    analyzer = SentimentIntensityAnalyzer()
    tweets['compound'] = [analyzer.polarity_scores(x)['compound'] for x in tweets['tweet']]
    tweets['rating'] = [1 if x > 0.05 else 0 if x > -0.05 else -1 for x in tweets['compound']]

    bins_neut = {
        '2020-02': 0,
        '2020-03': 0,
        '2020-04': 0,
        '2020-05': 0,
        '2020-06': 0,
        '2020-07': 0,
        '2020-08': 0,
        '2020-09': 0,
        '2020-10': 0,
        '2020-11': 0,
        '2020-12': 0,
        '2021-01': 0
    }
    bins_neg = bins_neut.copy()
    bins_pos = bins_neut.copy()
    for _, row in tweets.iterrows():
        dates = row['date'].split('-')
        date = dates[0] + '-' + dates[1]
        if row['rating'] == 1:
            bins_pos[date] += 1
        elif row['rating'] == 0:
            bins_neut[date] += 1
        else:
            bins_neg[date] += 1

    print(bins_pos)
    print(bins_neut)
    print(bins_neg)
    # ilosciowo
    # data = [bins_pos.values(),
    #         bins_neut.values(),
    #         bins_neg.values()]
    # procentowo
    print((100 * pd.DataFrame(bins_pos.values()) / (
                pd.DataFrame(bins_neut.values()) + pd.DataFrame(bins_neg.values())).to_numpy())[0])
    data = [
        (100 * pd.DataFrame(bins_pos.values()) / (
                    pd.DataFrame(bins_neut.values()) + pd.DataFrame(bins_neg.values())))[0],
        (100 * pd.DataFrame(bins_neut.values()) / (
                    pd.DataFrame(bins_pos.values()) + pd.DataFrame(bins_neg.values())))[0],
        (100 * pd.DataFrame(bins_neg.values()) / (
                    pd.DataFrame(bins_neut.values()) + pd.DataFrame(bins_pos.values())))[0]]
    X = np.arange(len(bins_pos))
    fig, ax = plt.subplots()
    ax.bar(X + 0.00, data[0], color='g', width=0.25)
    ax.bar(X + 0.25, data[1], color='b', width=0.25)
    ax.bar(X + 0.50, data[2], color='r', width=0.25)
    ax.set_title('Rozkład wyników w czasie')
    ind = np.arange(len(bins_pos))  # the x locations for the groups
    ax.set_xticks(ind, minor=False)
    ax.set_xticklabels(bins_pos.keys(), minor=False)
    ax.set_yticks(np.arange(0, 100, 10), minor=False)
    ax.legend(labels=['Pozytywne', 'Neutralne', 'Negatywne'])
    plt.show()

    tweets.to_csv('en/polarized-' + filename.split('/')[-1], index=False)

def word_clouds(filename):
    tweets = pd.read_csv(filename)
    print(len(tweets))

    #all
    lemmatized_tweets = []
    for index, row in tweets.iterrows():
        lemmatized_sentence = cleanup_sentence(pos_tag(word_tokenize(row['tweet'])))
        lemmatized_tweets.append(lemmatized_sentence)

    full_list = []
    for t in lemmatized_tweets:
        full_list += t

    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    # text = 'all your base are belong to us all of your base base base'
    wordcloud = WordCloud(width=1800, height=1400).generate(str(full_list))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    # sentiments
    neutral_tweets = []
    neg_tweets = []
    pos_tweets = []
    for index, row in tweets.iterrows():
        lemmatized_sentence = cleanup_sentence(pos_tag(word_tokenize(row['tweet'])))
        if row['rating'] == 0:
            neutral_tweets.append(lemmatized_sentence)
        elif row['rating'] == 1:
            pos_tweets.append(lemmatized_sentence)
        else:
            neg_tweets.append(lemmatized_sentence)

    full_list_neutral = []
    full_list_pos = []
    full_list_neg = []
    for t in neutral_tweets:
        full_list_neutral += t
    for t in pos_tweets:
        full_list_pos += t
    for t in neg_tweets:
        full_list_neg += t

    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    # text = 'all your base are belong to us all of your base base base'
    wordcloud = WordCloud(width=1800, height=1400).generate(str(full_list_neutral))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    # text = 'all your base are belong to us all of your base base base'
    wordcloud = WordCloud(width=1800, height=1400).generate(str(full_list_pos))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    # text = 'all your base are belong to us all of your base base base'
    wordcloud = WordCloud(width=1800, height=1400).generate(str(full_list_neg))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    # print('covid+szczepionka')
    # polish_tweets_analysis('en/covid+szczepionka-sentiment-2021-01-18.csv')
    # print('covid+szczepimysie')
    # polish_tweets_analysis('en/covid+szczepimysie-sentiment-2021-01-18.csv')
    # print('covid+szczepienie')
    # polish_tweets_analysis('en/covid+szczepienie-sentiment-2021-01-18.csv')
    # print('covid+szczepienie')
    # pl_freq_tweets_analysis('en/covid+szczepienie-sentiment-2021-01-18.csv')
    # print('covid+szczepimysie')
    # pl_freq_tweets_analysis('en/covid+szczepimysie-sentiment-2021-01-18.csv')
    # print('covid+szczepionka')
    # pl_freq_tweets_analysis('en/covid+szczepionka-sentiment-2021-01-18.csv')

    # print('covid+vaccine')
    # english_sentiment_tweets_analysis_vader('en/covid+vaccine-sentiment-2021-01-18.csv')

    # print('word_clouds')
    # word_clouds('en/polarized-covid+vaccine-sentiment-2021-01-18.csv')
    pass
