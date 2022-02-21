import pathlib
import re
import statistics
import sys

import langid
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import names, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd


name_words = set(names.words(["female.txt", "male.txt"]))
stop_words = set(stopwords.words("english"))


def is_recognized_name(text):
    return any(word in name_words for word in word_tokenize(text))


def is_nan_or_blank(text):
    return not isinstance(text, str) or text == ""


def is_single_word(text):
    return len(word_tokenize(text)) == 1


def contains_twitter_handle(text):
    return bool(re.search(r"\s(@[\w_-]+)", string=text))


def remove_twitter_handles(text):
    return re.sub(r"\s(@[\w_-]+)", repl="", string=text)


def remove_hashtags(text):
    return re.sub(r"\s(#[\w_-]+)", repl="", string=text)


def contains_hashtag(text):
    return bool(re.search(r"\s(#[\w_-]+)", string=text))


def in_english(text):
    return langid.classify(text)[0] == "en" 


def clean(text):
    return [_clean(sentence) for sentence in sent_tokenize(text)]


def _clean(text):
    wnl = WordNetLemmatizer()   
    text = remove_hashtags(text)
    text = remove_twitter_handles(text)
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word.isalpha()]
    words = [wnl.lemmatize(word) for word in words]

    return words


def analyze_sentiments(sentences):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    sentiments = []
    for sentence in sentences:
        sentence = " ".join(sentence)
        sentiment = sentiment_analyzer.polarity_scores(sentence)
        sentiments.append(sentiment["compound"])

    return std_dev_or_na(sentiments)


def std_dev_or_na(items):
    try:
        std_dev = statistics.stdev(items)
    except statistics.StatisticsError:
        return np.nan
    else:
        return std_dev


def count_unique_words(sentences):
    return len(set(flatten_nested_list(sentences)))


def sentence_length_deviation(sentences):
    return std_dev_or_na([len(sent) for sent in sentences])


def flatten_nested_list(data):
    return [item for sublist in data for item in sublist]


def feature_selection(df):
    # drop rows with NA `text` and replace NA `author` with blank string
    df = df.dropna(subset=["text"])
    df["author"] = df["author"].fillna("")

    # detect article langauge and drop non-english rows
    df["language_english"] = df["text"].map(in_english)
    df = df.loc[df["language_english"] == True]

    # add boolean features and convert True/False to 1/0
    df["has_twitter_handle"] = df["text"].map(contains_twitter_handle)
    df["has_twitter_handle"] = df["has_twitter_handle"].astype(int)

    df["has_hashtag"] = df["text"].map(contains_hashtag)
    df["has_hashtag"] = df["has_hashtag"].astype(int)

    df["author_name_single_word"] = df["author"].map(is_single_word)
    df["author_name_single_word"] = df["author_name_single_word"].astype(int)

    df["author_name_recognized"] = df["author"].map(is_recognized_name)
    df["author_name_recognized"] = df["author_name_recognized"].astype(int)

    df["author_name_is_na"] = df["author"].map(is_nan_or_blank)
    df["author_name_is_na"] = df["author_name_is_na"].astype(int)

    # tokenize text into sentences and words (list of sentences containing lists of words)
    df["sentences"] = df["text"].map(clean)

    # add numeric features
    df["sentiment_consistency"] = df["sentences"].map(analyze_sentiments)
    df["sentence_length_consistency"] = df["sentences"].map(sentence_length_deviation)
    df["unique_word_count"] = df["sentences"].map(count_unique_words)

    # drop original columns and save to file
    df = df.drop(columns=["title", "text", "sentences", "author", "language_english"])
    
    return df


def plot_author_features(df):
    df_real = df.loc[df["label"] == 0]
    df_fake = df.loc[df["label"] == 1]

    author_recognized_real = len(df_real.loc[df_real["author_name_recognized"] == 1]) / len(df_real)
    author_recognized_fake = len(df_fake.loc[df_fake["author_name_recognized"] == 0]) / len(df_fake)

    author_single_real = len(df_real.loc[df_real["author_name_single_word"] == 1]) / len(df_real)
    author_single_fake = len(df_fake.loc[df_fake["author_name_single_word"] == 0]) / len(df_fake)

    author_missing_real = len(df_real.loc[df_real["author_name_is_na"] == 1]) / len(df_real)
    author_missing_fake = len(df_fake.loc[df_fake["author_name_is_na"] == 0]) / len(df_fake)

    real_news = [author_recognized_real, author_single_real, author_missing_real]
    fake_news = [author_recognized_fake, author_single_fake, author_missing_fake]

    bar_width = 0.25

    br1 = np.arange(3)
    br2 = [x + bar_width for x in br1]

    plt.bar(br1, real_news, width = bar_width, label ='Real', alpha=0.75)
    plt.bar(br2, fake_news, width = bar_width, label ='Fake', alpha=0.75)

    plt.title("Author features")
    plt.ylabel('Percentage')
    plt.grid(axis="y", alpha=0.5)
    plt.xticks(ticks=[0.25, 1.25, 2.25], labels=["Author recognized", "Author single word", "Author missing"])
 
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig("figs/author_features.png", bbox_inches='tight')


def plot_content_features(df):
    df_real = df.loc[df["label"] == 0]
    df_fake = df.loc[df["label"] == 1]

    twitter_handles_real = len(df_real.loc[df_real["has_twitter_handle"] == 1]) / len(df_real)
    twitter_handles_fake = len(df_fake.loc[df_fake["has_twitter_handle"] == 0]) / len(df_fake)

    hashtags_real = len(df_real.loc[df_real["has_hashtag"] == 1]) / len(df_real)
    hashtags_fake = len(df_fake.loc[df_fake["has_hashtag"] == 0]) / len(df_fake)

    real_news = [twitter_handles_real, hashtags_real]
    fake_news = [twitter_handles_fake, hashtags_fake]

    bar_width = 0.25
 
    br1 = np.arange(2)
    br2 = [x + bar_width for x in br1]

    plt.close()
    plt.bar(br1, real_news, width=bar_width, label='Real', alpha=0.75)
    plt.bar(br2, fake_news, width=bar_width, label='Fake', alpha=0.75)

    plt.ylabel('Percentage')
    plt.grid(axis="y", alpha=0.5)
    plt.xticks(ticks=[0.25, 1.25], labels=["Contains Twitter handles", "Contains hashtags"])
 
    plt.title("Content features")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig("figs/content_features.png", bbox_inches='tight')


def plot_sentiment(df):
    plt.close()
    plt.hist(df.loc[df["label"] == 0, "sentiment_consistency"], bins=70, alpha=0.75, label="Real")
    plt.hist(df.loc[df["label"] == 1, "sentiment_consistency"], bins=70, alpha=0.75, label="Fake")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title("Within-article sentiment variance")
    plt.xlabel("Frequency")
    plt.ylabel("Variance of sentiment within article")
    plt.grid(axis="y", alpha=0.5)
    plt.savefig("figs/sentiment.png", bbox_inches='tight')


def plot_unique_words(df):
    plt.close()
    plt.hist(df.loc[df["label"] == 0, "unique_word_count"], bins=70, alpha=0.75, label="Real")
    plt.hist(df.loc[df["label"] == 1, "unique_word_count"], bins=70, alpha=0.75, label="Fake")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title("Unique word counts")
    plt.xlabel("Frequency")
    plt.ylabel("Number of unique words")
    plt.grid(axis="y", alpha=0.5)
    plt.xlim(right=2000)
    plt.savefig("figs/unique_words.png", bbox_inches='tight')


if __name__ == "__main__":
    nltk.download(["names", "omw-1.4", "punkt", "stopwords", "vader_lexicon", "wordnet"])
    
    figs_path = pathlib.Path.cwd() / "figs"
    figs_path.mkdir(exist_ok=True)

    data_path = pathlib.Path.cwd() / "data"
    test_path = data_path / "test.csv"
    train_path = data_path / "train.csv"

    if not test_path.exists():
        raise sys.exit(f"Cannot locate testing data: {test_path}.\n\nPlease ensure this file exists and try again.")
    if not train_path.exists():
        raise sys.exit(f"Cannot locate training data: {train_path}.\n\nPlease ensure this file exists and try again.")

    df_train = pd.read_csv(train_path)
    df_train = feature_selection(test_path)
    train_out_path = data_path / "train_features.csv"
    df_train.to_csv(train_out_path)

    df_test = pd.read_csv("data/test.csv")
    df_test = feature_selection(df_test)
    test_out_path = data_path / "test_features.csv"
    df_test.to_csv(test_out_path)

    plot_author_features(df_train)
    plot_content_features(df_train)
    plot_sentiment(df_train)
    plot_unique_words(df_train)
