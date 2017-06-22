"""
Feed training data to classifier
"""

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
from sklearn.linear_model import LogisticRegression


logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
TOKER = RegexpTokenizer(r'\w+')
WNL = WordNetLemmatizer()
TFIDF = TfidfTransformer(norm='l2', sublinear_tf=False, use_idf=True)

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

job_queue = ['ag_news_csv', 'dbpedia_csv', 'yahoo_answers_csv',
             'amazon_review_full_csv', 'amazon_review_polarity_csv',
             'yelp_review_polarity_csv', 'yelp_review_full_csv']

# Try classifier
def worker(dataset):
    logging.info('Dataset: {}'.format(dataset))
    logfile = 'result_classify.log'
    h = open(logfile, 'a')
    # folder = '../DATA/TextClassificationDatasets/'
    try:
        ## freq
        m, X_train, y_train, X_test, y_test = pickle.load(open(dataset + '_freq_traindata.p','r'))
        # count to tfidf
        X_train = TFIDF.fit_transform(X_train)
        X_test = TFIDF.fit_transform(X_test)
        # clf_descr, train_accu, test_accu, train_time = benchmark(LinearSVC(loss='hinge', penalty='l2'), X_train,
        #                                                          y_train, X_test, y_test)
        clf_descr, train_accu, test_accu, train_time = benchmark(LogisticRegression(penalty='l2', n_jobs=4), X_train,
                                                                 y_train, X_test, y_test)
        #
        logging.info('{}\t{}\ttrain_accu:{}\ttest_accu:{}'.format(dataset+'_freq', clf_descr, train_accu, test_accu))
        #
        h.write('{}\t{}\ttrain_accu:{}\ttest_accu:{}\t#features:{}\n'.format(dataset + '_freq', clf_descr, train_accu,
                                                                             test_accu, len(m)))
        del m, X_train, y_train, X_test, y_test
        ## PMI
        # m, X_train, y_train, X_test, y_test = pickle.load(open(dataset + '_wpmi_traindata.p', 'r'))
        # # count to tfidf
        # X_train = TFIDF.fit_transform(X_train)
        # X_test = TFIDF.fit_transform(X_test)
        # clf_descr, train_accu, test_accu, train_time = benchmark(LinearSVC(loss='hinge', penalty='l2'), X_train,
        #                                                          y_train, X_test, y_test)
        # h.write('{}\t{}\ttrain_accu:{}\ttest_accu:{}\t#features:{}\n'.format(dataset + '_wpmi', clf_descr, train_accu,
        #                                                                      test_accu, len(m)))
        # clf_descr, train_accu, test_accu, train_time = benchmark(LogisticRegression(penalty='l2', n_jobs=4), X_train,
        #                                                          y_train, X_test, y_test)
        # h.write('{}\t{}\ttrain_accu:{}\ttest_accu:{}\t#features:{}\n'.format(dataset + '_wpmi', clf_descr, train_accu,
        #                                                                      test_accu, len(m)))
        # del m, X_train, y_train, X_test, y_test
        # return dataset, clf_descr, train_accu, test_accu, train_time
    except:
        logging.info('{}----failed----'.format(dataset))
    h.close()


# worker(job_queue[0])

multiprocessing.Pool(8).map(worker, job_queue)

# dataset = job_queue[3]
# worker(dataset)

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