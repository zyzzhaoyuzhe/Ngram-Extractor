import logging
from time import time
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from Queue import Queue
import matplotlib.pyplot as plt
import threading, multiprocessing
from time import sleep
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
import lib_ngram
import math, operator

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
toker = RegexpTokenizer(r'\w+')
wnl = WordNetLemmatizer()
tfidf = TfidfTransformer(norm='l2', sublinear_tf=False, use_idf=True)

class Read_Zhang():
    def __init__(self, file):
        self.file = file
    def label_body(self, ncol=3):
        h = open(self.file, 'r')
        h.seek(0)
        for line in h:
            line = line.split('","', ncol-1)
            label = int(line[0][1:])
            body = map(lambda s: s.decode('utf-8').lower(), line[1:ncol-1]) + sent_tokenize(line[-1].decode('utf-8').lower())
            body = map(toker.tokenize, body)
            yield label, body
    def body(self, ncol=3):
        gen = self.label_body(ncol=ncol)
        while True:
            try:
                _, foo = gen.next()
                for sent in foo:
                    yield sent
            except StopIteration:
                raise StopIteration

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    #
    t0 = time()
    pred = clf.predict(X_train)
    score_time = time() - t0
    print("score time:  %0.3fs" % score_time)
    train_accu = metrics.accuracy_score(y_train, pred)
    print('Train accuracy:   %0.3f; Error rate  %0.3f' % (train_accu, 1-train_accu))
    #
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    test_accu = metrics.accuracy_score(y_test, pred)
    print('Test accuracy:   %0.3f; Error rate   %0.3f' % (test_accu, 1-test_accu))
    clf_descr = str(clf).split('(')[0]
    return clf_descr, train_accu, test_accu, train_time



if __name__ == '__main__':
    print ' '
    print ' '

    job_queue = [('ag_news_csv',3), ('amazon_review_full_csv',3), ('dbpedia_csv', 3), ('yahoo_answers_csv', 4), ('yelp_review_polarity_csv',2), ('yelp_review_full_csv',2)]
    topN = [100000, 500000]

    def _worker(item):
        output = []
        dataset, ncol = item
        logging.info('Dataset: {}'.format(dataset))
        folder = '/media/vincent/Data-adhoc/TextClassificationDatasets/'
        train_file = folder + dataset + '/train.csv'
        test_file = folder + dataset + '/test.csv'
        text_train = Read_Zhang(train_file)
        text_test = Read_Zhang(test_file)
        # feature extractor
        feature_freq = lib_ngram.BOW_freq(5)
        feature_freq.get_ngram(text_train.body(ncol=ncol), [100000, 500000], mode='s')
        feature_wpmi = lib_ngram.BOW_wpmi(5)
        feature_wpmi.get_ngram(text_train.body(ncol=ncol), [100000, 500000])
        # train data and test data (freq ngram)
        logging.info('------Using freq ngram features------')
        X_train, y_train = feature_freq.raw_count(text_train.label_body(ncol=ncol))
        X_test, y_test = feature_freq.raw_count(text_test.label_body(ncol=ncol))
        # tfidf
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.fit_transform(X_test)
        # benchmark(MultinomialNB())
        clf_descr, train_accu, test_accu, train_time = benchmark(LinearSVC())
        output.append((dataset + 'freq', topN, clf_descr, train_accu, test_accu, train_time))
        # train data and test data (wpmi ngram)
        logging.info('------Using WPMI ngram features------')
        X_train, y_train = feature_wpmi.raw_count(text_train.label_body(ncol=ncol))
        X_test, y_test = feature_wpmi.raw_count(text_test.label_body(ncol=ncol))
        # tfidf
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.fit_transform(X_test)
        # benchmark(MultinomialNB())
        clf_descr, train_accu, test_accu, train_time = benchmark(LinearSVC())
        output.append((dataset + 'wpmi', topN, clf_descr, train_accu, test_accu, train_time))
        return output

    def worker_loop(item):
        # try:
        return _worker(item)
            # dataset, ncol = item
            # return [(dataset + '/freq', [100,1], 'svm', 0.8, 0.8, 2),
            #         (dataset + '/wpmi', [100,1], 'svm', .8, .9, 1)]
        # except:
        #     logging.info('************ Fail ************* {}'.format(item[0]))

    item = job_queue[1]
    worker_loop(item)

    # jobs = []
    # nproc = 2
    #
    # result = multiprocessing.Pool(nproc).map(worker_loop, job_queue)
    # foo = []
    # for item in result:
    #     if item:
    #         foo += item
    # result = foo
    # # display results
    # indices = np.arange(len(result))
    # descrip, train_accu, test_accu = [[x[i] for x in result] for i in [0,3,4]]
    # plt.figure()
    # plt.barh(indices, train_accu, .3, label='train_accu', color='navy')
    # plt.barh(indices + 0.4, test_accu, .3, label='test_accu', color='c')
    # plt.yticks(())
    # for i, c in zip(indices, descrip):
    #     plt.text(-.1, i, c)
    #
    # print 'done'