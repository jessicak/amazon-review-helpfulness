from __future__ import division, print_function, absolute_import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import nltk
import numpy as np
import pandas as pd
import random
import sklearn
import string


class Dataset(object):
    columns = ['reviewer_id', 'asin', 'review_text', 'overall', 'category', 'good', 'bad']

    def __init__(self, path, we_model, params):
        """Construct a Dataset object.

        @param path - A string denoting the location of the dataset CSV file
        @param we_model - The trained word embedding model (e.g., Word2vec)
        @param params - A dictionary of optional parameters

        params (default values):
        {
            'label' : 'good',   # Column on which one-hot label encoding should be performed
            'max_size' : 200,   # Maximum number of words to be encoded per review
            'max_tfidf': 150    # Maximum number of tf-idf features
            'n_features' : 300, # Length of a single word embedding
            'test_size' : 0.25, # Proportion of data that should be allocated for testing
            'verbose' : False   # Print messages to console to track constructor progress
        }

        """
        self.verbose = params.get('verbose', False)
        self.test_size = params.get('test_size', 0.25)
        self.we_model = we_model
        self.label = params.get('label', 'good')

        self.__cond_print("Initializing dataset at " + path + ".")
        self.df = pd.read_csv(path, sep='|', index_col=0)

        self.__cond_print("Splitting training and test sets.")
        self.__split_train_test()

        # Build indexing and validate columns
        self.index = {}
        for i, col_name in enumerate(self.df.columns.values):
            self.index[col_name] = i + 1
        for col_name in Dataset.columns:
            assert col_name in self.index

        # Build reviewer/ASIN maps
        self.__cond_print("Building reviewer/ASIN maps.")
        self.reviewer_map, self.asin_map = {}, {}
        for entry in self.train.itertuples():
            rid, pid = entry[self.index['reviewer_id']], entry[self.index['asin']]
            self.reviewer_map[rid] = self.reviewer_map.get(rid, 0) + 1
            self.asin_map[pid] = self.asin_map.get(pid, 0) + 1

        # Train tf-idf featurizer
        self.__cond_print("Fitting tf-idf featurizer.")
        self.max_tfidf = params.get('max_tfidf', 150)
        if self.max_tfidf < 1:
            self.tfidf = None
            self.tfidf_train = None
            self.tfidf_test = None
        else:
            self.tfidf = TfidfFeaturizer(col_name='review_text', max_features=self.max_tfidf)
            self.tfidf.fit_transform(self.df)
            self.tfidf_train = self.tfidf.transform(self.train)
            self.tfidf_test = self.tfidf.transform(self.test)

        # Tune tf-idf logistic regression
        self.__cond_print("Tuning tf-idf classifier.")
        if self.max_tfidf < 1:
            self.tfidf_classifier = None
        else:
            self.tfidf_classifier = TfidfClassifier(self.tfidf_train, 
                np.array(self.train[self.label]))
        
        self.__cond_print("Featurizing training set.")
        training_am = self.asin_map.copy()
        training_rm = self.reviewer_map.copy()
        for key in training_am:
            training_am[key] -= 1
        for key in training_rm:
            training_rm[key] -= 1
        params['asin_map'] = training_am
        params['reviewer_map'] = training_rm
        params['tfidf_matrix'] = self.tfidf_train
        params['tfidf_classifier'] = self.tfidf_classifier
        self.training_data = ReviewSequenceData(self.train, self.index, we_model, params)

        self.__cond_print("Featurizing test set.")
        params['asin_map'] = self.asin_map
        params['reviewer_map'] = self.reviewer_map
        params['tfidf_matrix'] = self.tfidf_test
        self.test_data = ReviewSequenceData(self.test, self.index, we_model, params)

        self.__cond_print("Finished configuring dataset.")

    def next(self, batch_size):
        return self.training_data.next(batch_size)

    def get_test_batch(self, batch_size=-1):
        if batch_size < 0:
            return self.test_data.next(self.test_data.size())
        else:
            return self.test_data.next(batch_size)

    def get_n_cols(self):
        return self.training_data.get_n_cols()

    def __split_train_test(self):
        train, test = train_test_split(self.df, test_size=self.test_size, 
                                       random_state=42)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        self.train, self.test = train, test

    def __cond_print(self, s):
        if self.verbose:
            print(s)


class ReviewSequenceData(object):
    def __init__(self, df, col_index, we_model, params):
        # Optional parameter extraction
        self.reviewer_map = params.get('reviewer_map', {})
        self.asin_map = params.get('asin_map', {})
        self.n_features = params.get('n_features', we_model['food'].shape[0])
        self.max_size = params.get('max_size', 200)
        self.label = params.get('label', 'good')
        self.tfidf_matrix = params.get('tfidf_matrix', None)
        self.tfidf_classifier = params.get('tfidf_classifier', None)

        self.index = col_index
        self.we_model = we_model
        self.data, self.labels, self.seqlen = [], [], []
        self.batch_id = 0
        self.n_reviews = len(df)
        self.dnn_data = None

        for i, row in enumerate(df.itertuples()):
            # LSTM featurization
            embed, lab, sl, n_skip = self.__encode_lstm_features(row)
            self.data.append(embed)
            self.labels.append(lab)
            self.seqlen.append(sl)
            encode_ratio = (1.0 * sl) / (sl + n_skip)

            # DNN featurization
            if self.dnn_data is None:
                dnn_row = self.__encode_dnn_features(i, row, encode_ratio)
                self.n_cols = dnn_row.shape[0]
                self.dnn_data = np.zeros([self.n_reviews, self.n_cols], dtype='float32')
                self.dnn_data[i, :] = dnn_row
            else:
                self.dnn_data[i, :] = self.__encode_dnn_features(i, row, encode_ratio)

    def __label_one_hot(self, row):
        if self.label == 'good':
            if row[self.index[self.label]]:
                return np.array([1., 0.])
            else:
                return np.array([0., 1.])
        if self.label == 'bad':
            if row[self.index[self.label]]:
                return np.array([0., 1.])
            else:
                return np.array([1., 0.])

    def __encode_lstm_features(self, row):
            words = row[self.index['review_text']].split()
            embedded_review = np.zeros([self.max_size, self.n_features], 
                                       dtype='float32')
            j = 0
            skip_count = 0
            for word in words:
                if word in self.we_model:
                    embedded_review[j, :] = self.we_model[word]
                    j += 1
                elif word.capitalize() in self.we_model:
                    embedded_review[j, :] = self.we_model[word.capitalize()]
                    j += 1
                else:
                    skip_count += 1
                if j == self.max_size:
                    break
            return embedded_review, self.__label_one_hot(row), j, skip_count

    def __encode_dnn_features(self, i, row, encode_ratio):
        category_onehot = [0.0] * 24
        category_onehot[int(row[self.index['category']])] = 1.0
        overall_onehot = [0.0] * 5
        overall_onehot[int(row[self.index['overall']]) - 1] = 1.0
        asin_count = self.asin_map.get(row[self.index['asin']], 0) + 1
        r_count = self.reviewer_map.get(row[self.index['reviewer_id']], 0) + 1
        f1 = np.array([asin_count, r_count, encode_ratio] + category_onehot + overall_onehot)
        f2 = TextFeaturizer.featurize(row[self.index['review_text']])
        if self.tfidf_matrix is not None and self.tfidf_classifier is not None:
            f3 = self.tfidf_classifier.predict_probs(self.tfidf_matrix[i, :])
        else:
            f3 = np.array([])
        return np.hstack([f1, f2, f3])

    def size(self):
        return self.n_reviews

    def get_n_cols(self):
        return self.n_cols

    def next(self, batch_size):
        data_len = len(self.data)
        if self.batch_id == data_len:
            self.batch_id = 0
        end_idx = min(self.batch_id + batch_size, data_len)
        batch_data = (self.data[self.batch_id:end_idx])
        batch_labels = self.labels[self.batch_id:end_idx]
        batch_seqlen = self.seqlen[self.batch_id:end_idx]
        batch_dnn = self.dnn_data[self.batch_id:end_idx, :]
        self.batch_id = end_idx
        return np.stack(batch_data), batch_labels, batch_seqlen, batch_dnn


class TfidfFeaturizer(object):
    def __init__(self, col_name='review_text', max_features=150, should_stem=False):
        self.col_name = col_name
        self.max_features = max_features
        self.stemmer = nltk.stem.lancaster.LancasterStemmer()
        self.vectorizer = None
        self.should_stem = should_stem

    def fit_transform(self, df):
        docs = self.__create_doc_list(df)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1,3),
            max_features=self.max_features
        )
        return (self.vectorizer.fit_transform(docs)).toarray()

    def transform(self, df):
        if self.vectorizer is not None:
            docs = self.__create_doc_list(df)
            return (self.vectorizer.transform(docs)).toarray()
        else:
            return None

    def __create_doc_list(self, df):
        X = df[self.col_name].tolist()
        if self.should_stem:
            X_stem = [None] * len(X)
            for i in xrange(len(X)):
                X_stem[i] = ' '.join([self.stemmer.stem(w) for w in X[i].split()])
            return X_stem
        else:
            return X


class TextFeaturizer(object):
    # universal_pos = [u'ADJ', u'ADP', u'ADV', u'CONJ', u'DET', u'NOUN', u'NUM', 
    #     u'PRT', u'PRON', u'VERB', u'.', u'X']

    universal_pos = [u'ADJ', u'ADP', u'ADV', u'CONJ', u'NOUN', u'VERB', u'X']

    @staticmethod
    def featurize(text):
        features = []
        words = text.split()

        # word count
        wc = len(words)
        features.append(wc)

        # mean/std word length
        word_lens = [len(w) for w in words]
        features.append(np.mean(word_lens))
        features.append(np.std(word_lens))

        # text length
        features.append(np.sum(word_lens))

        # POS distribution
        tagged_pos = [t[1] for t in nltk.pos_tag(words, tagset='universal')] # SLOW
        pd = { pos : 0.0 for pos in TextFeaturizer.universal_pos }
        n_tagged = 0.0
        for pos in tagged_pos:
            if pos in pd:
                pd[pos] += 1.0
                n_tagged += 1.0
        for pos in TextFeaturizer.universal_pos:
            features.append(pd[pos] / n_tagged)
        return np.array(features)
    
    @staticmethod
    def feature_length():
        return len(TextFeaturizer.featurize('sample sentence'))


class TfidfClassifier(object):
    def __init__(self, tfidf_matrix, labels):
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(tfidf_matrix, labels, 
                test_size=0.25, random_state=42)
        self.c = self.__find_optimal_c()
        self.lr = LogisticRegression(penalty='l2', C=self.c, n_jobs=-1)
        self.lr.fit(tfidf_matrix, labels)

    def predict_probs(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
            return np.squeeze(self.lr.predict_proba(X))
        else:
            return self.lr.predict_proba(X)

    def __make_rand_range(self, start, end, n):
        rr = []
        for _ in xrange(n):
            rr.append(start + (abs(end - start) * random.random()))
        return sorted(rr)

    def __find_optimal_c(self):
        def find_best_c(candidates):
            best_val_acc = -1.0
            best_c = None
            best_c_idx = None
            for idx, c in enumerate(candidates):
                lr = LogisticRegression(penalty='l2', C=c, n_jobs=-1)
                lr.fit(self.X_train, self.y_train)
                preds = lr.predict(self.X_val)
                acc = np.mean(preds == self.y_val)
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_c = c
                    best_c_idx = idx
            if best_c_idx == 0:
                start = candidates[0]
                end = candidates[2]
            elif best_c_idx == len(candidates) - 1:
                start = candidates[-3]
                end = candidates[-1]
            else:
                start = candidates[best_c_idx - 1]
                end = candidates[best_c_idx + 1]
            return best_c, best_val_acc, start, end 

        reg_str = [0.01, 0.1, 1.0, 10, 25, 50, 100, 500, 1000]
        overall_best_c = None
        overall_best_val_acc = -1
        for _ in xrange(4):
            best_c, best_val_acc, start, end = find_best_c(reg_str)
            if best_val_acc > overall_best_val_acc:
                overall_best_val_acc = best_val_acc
                overall_best_c = best_c
            reg_str = self.__make_rand_range(start, end, 25)
        return overall_best_c
