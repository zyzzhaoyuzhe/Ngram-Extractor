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
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
toker = RegexpTokenizer(r'\w+')
wnl = WordNetLemmatizer()
tfidf = TfidfTransformer(norm='l2', sublinear_tf=False, use_idf=True)

def benchmark(clf, X_train, y_train, X_test, y_test):
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

# if __name__ == '__main__':
print ' '
print ' '

job_queue = ['ag_news_csv', 'dbpedia_csv', 'yelp_review_polarity_csv', 'amazon_review_full_csv', 'yahoo_answers_csv', 'yelp_review_full_csv']
topN = [100000, 500000]
maxmem = (500000, 10000000)


def worker_loop(dataset):
    logging.info('Dataset: {}'.format(dataset))
    folder = '/media/vincent/Data-adhoc/TextClassificationDatasets/'
    try:
        model = lib_ngram.BOW_wpmi(5)
        model.load(dataset + '_wpmi.p')
        X_train, y_train = model.raw_count(folder + dataset + '/train_parsed.csv')
        X_test, y_test = model.raw_count(folder + dataset + '/test_parsed.csv')
        clf_descr, train_accu, test_accu, train_time = benchmark(LinearSVC(), X_train, y_train, X_test, y_test)
        return dataset, train_accu, test_accu, train_time
    except:
        logging.info('{}----failed----'.format(dataset))


dataset = job_queue[3]
worker_loop(dataset)

    # jobs = []
    # nproc = 4
    # result = multiprocessing.Pool(nproc).map(worker_loop, zip(job_queue, [topN]*len(job_queue)))
    # pickle.dump(result, open('result.p','w'))
    #
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