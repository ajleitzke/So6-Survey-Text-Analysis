import re
from collections import Counter
import pandas as pd
from nltk.stem import WordNetLemmatizer
import os

url = os.environ.get("url")

survey_df = pd.read_csv(
    'https://docs.google.com/spreadsheets/d/e/2PACX-1vTSi45lc_B_sFsPEqSqceMAdU42sZaEH_zc5wH_tmtTK0ZPN0dnP3J5u0F1XSAz5-ui35q4WaagBt1o/pub?gid=0&single=true&output=csv')
stop_words_file = 'stopwords.txt'
stop_words = []

with open(stop_words_file, "r") as f:
    for line in f:
        stop_words.extend(line.split())

stop_words = stop_words

def preprocess(raw_text):
    letters_only_text = re.sub("[^a-zA-Z]", " ", str(raw_text))

    words = letters_only_text.lower().split()

    cleaned_words = []

    for word in words:
        if word not in stop_words:
            cleaned_words.append(word)

    stemmed_words = []
    for word in cleaned_words:
        word = WordNetLemmatizer().lemmatize(word)
        stemmed_words.append(word)

    return " ".join(stemmed_words)


survey_df['prep-positive'] = survey_df['Positive'].apply(preprocess)
survey_df['prep-negative'] = survey_df['Negative'].apply(preprocess)
survey_df['prep-want'] = survey_df['Short-Term Want'].apply(preprocess)

positive_df = pd.DataFrame(Counter(" ".join(survey_df["prep-positive"]).split()).most_common(15))
positive_df.rename(columns={0: 'word', 1: 'count'}, inplace=True)
positive_df.to_csv('positive.csv', index=False)

negative_df = pd.DataFrame(Counter(" ".join(survey_df["prep-negative"]).split()).most_common(15))
negative_df.rename(columns={0: 'word', 1: 'count'}, inplace=True)
negative_df.to_csv('negative.csv', index=False)

want_df = pd.DataFrame(Counter(" ".join(survey_df["prep-want"]).split()).most_common(10))
want_df.rename(columns={0: 'word', 1: 'count'}, inplace=True)
want_df.to_csv('want.csv', index=False)

print(positive_df, negative_df, want_df)
