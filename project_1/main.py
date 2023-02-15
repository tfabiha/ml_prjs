import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np

import sklearn.naive_bayes
import features

def extract(train_path, test_path):
    data_train = features.extract(train_path)
    data_test = features.extract(test_path)

    return data_train, data_test

def Ngram(data_train, data_test):
    # NGram

    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')

    x_train = vectorizer.fit_transform(data_train['tokens'])
    x_test = vectorizer.transform(data_test['tokens'])

    y_train = data_train['label']
    y_test = data_test['label']
    
    selector = SelectKBest(score_func=f_classif,k=1000)
    x_train = selector.fit_transform(x_train,y_train)
    x_test = selector.transform(x_test)

    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    print('Classification report:')
    print(classification_report(y_test, y_predict, target_names=['negative', 'positive', 'neutral']))

def NgramLex(data_train, data_test, lex_path):
    # NGram + Lex

    lex_features_train = features.get_lexHS_features(data_train['tokens'], lex_path)
    lex_features_test = features.get_lexHS_features(data_test['tokens'], lex_path)

    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
    x_train = vectorizer.fit_transform(data_train['tokens'])
    x_test = vectorizer.transform(data_test['tokens'])

    y_train = data_train['label']
    y_test = data_test['label']
    
    selector = SelectKBest(score_func=f_classif,k=1000)
    x_train = selector.fit_transform(x_train,y_train)
    x_test = selector.transform(x_test)

    x_train = hstack((x_train, lex_features_train))
    x_test = hstack((x_test, lex_features_test))

    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    print('Classification report:')
    print(classification_report(y_test, y_predict, target_names=['negative', 'positive', 'neutral']))

def NgramLexEnc(data_train, data_test, lex_path):
    # NGram + Lex + Encoding

    lex_features_train = features.get_lexHS_features(data_train['tokens'], lex_path)
    lex_features_test = features.get_lexHS_features(data_test['tokens'], lex_path)

    enc_features_train = features.get_encoding_features(data_train['tokens'], data_train["pos_tags"])
    enc_features_test = features.get_encoding_features(data_test['tokens'], data_test["pos_tags"])

    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
    x_train = vectorizer.fit_transform(data_train['tokens'])
    x_test = vectorizer.transform(data_test['tokens'])

    y_train = data_train['label']
    y_test = data_test['label']
    
    selector = SelectKBest(score_func=f_classif,k=1000)
    x_train = selector.fit_transform(x_train,y_train)
    x_test = selector.transform(x_test)

    x_train = hstack((x_train, lex_features_train))
    x_train = hstack((x_train, enc_features_train))
    x_test = hstack((x_test, lex_features_test))
    x_test = hstack((x_test, enc_features_test))

    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    print('Classification report:')
    print(classification_report(y_test, y_predict, target_names=['negative', 'positive', 'neutral']))

def Custom(data_train, data_test, lex_path):
    # Custom - NGram + POS

    lex_HS_features_train = features.get_lexHS_features(data_train['tokens'], lex_path)
    lex_Sent_features_train = features.get_lexSent_features(data_train['tokens'], lex_path)
    lex_HS_features_test = features.get_lexHS_features(data_test['tokens'], lex_path)
    lex_Sent_features_test = features.get_lexSent_features(data_test['tokens'], lex_path)

    #enc_features_train = features.get_encoding_features(data_train['tokens'], data_train["pos_tags"])
    #enc_features_test = features.get_encoding_features(data_test['tokens'], data_test["pos_tags"])

    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
    x_train = vectorizer.fit_transform(data_train['tokens'])
    x_test = vectorizer.transform(data_test['tokens'])

    y_train = data_train['label']
    y_test = data_test['label']
    
    selector = SelectKBest(score_func=f_classif,k=1000)
    x_train = selector.fit_transform(x_train,y_train)
    x_test = selector.transform(x_test)

    x_train = hstack((x_train, lex_HS_features_train))
    x_train = hstack((x_train, lex_Sent_features_train))
    x_test = hstack((x_test, lex_HS_features_test))
    x_test = hstack((x_test, lex_Sent_features_test))

    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    print('Classification report:')
    print(classification_report(y_test, y_predict, target_names=['negative', 'positive', 'neutral']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Enc", "Custom"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    args = parser.parse_args()


    ## example access parsed args
    ## print(args.lexicon_path)

    data_train, data_test = extract(args.train, args.test)
    
    if args.model == "Ngram":
        Ngram(data_train, data_test)
    elif args.model == "Ngram+Lex":
        NgramLex(data_train, data_test, args.lexicon_path)
    elif args.model == "Ngram+Lex+Enc":
        NgramLexEnc(data_train, data_test, args.lexicon_path)
    elif args.model == "Custom":
        Custom(data_train, data_test, args.lexicon_path)

    """
    Performance
    | Models        | macro F1 | negative F1 | positive F1 | neutral F1 |
    |---------------|----------|-------------|-------------|------------|
    | Ngram         | 0.44     | 0.11        | 0.53        | 0.68       |
    | Ngram+Lex     | 0.52     | 0.34        | 0.56        | 0.65       |
    | Ngram+Lex+Enc | 0.50     | 0.36        | 0.52        | 0.63       |
    | Custom        | 0.52     | 0.40        | 0.54        | 0.61       |
    
    I encoded POS tags and All Caps in the Ngram+Lex+Enc model. POS tags was chosen because of all the 
    encodings Mohammad et al. shows that dropping POS reduces F1-score by 0.64. This is greater than the 
    contributions of all the other encodings together, which is 0.14. Of the other encodings I chose 
    All Caps because this encoding is usually used to depict excitement so I thought it would help determine 
    the difference between subjective and objective tweets more.
    
    Ngram+Lex performed the best, with an F1-score of 0.52. The Ngram model only had an F1-score of 0.44 and 
    the Ngram+Lex+Enc model only had an F1-score of 0.50. This is probably because the the Ngram+Lex model was 
    able to add necessary features to identify sentiment instead of simply overfitting the data, which is a 
    concern every time we add any new features. The Ngram+Lex+Enc model didn’t add much that the Ngram+Lex 
    model didn’t have so the extra features just helped to overfit the data with the training set, meaning 
    that the Ngram+Lex model would be perform better at identifying sentiment since it isn’t as overfit on the 
    training data as the Ngram+Lex+Enc model.

    I added in the Sentiment Lexicon to the Ngram+Lex model, since that model had the best macro F1-score and 
    Mohammad et al. shows that lexicons contribute a lot to the F1-score, I thought it would be best to have 
    both lexicons evaluating the tokens of the tweets and that this might improve the overally quality. This 
    did improve the performance of the Ngram+Lex model. Specifically, the F1-score of identifying a negative 
    tweet went from 0.34 to 0.40, but the macro F1-score stayed at 0.52.
    """
