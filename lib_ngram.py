import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import numpy as np
import scipy.sparse as sps
from collections import defaultdict
import cPickle as pickle
from nltk.tokenize import word_tokenize
import operator
from itertools import chain, tee
import math
from sys import getsizeof
from guppy import hpy
logger = logging.getLogger()
logger.setLevel(20)

class smartfile(object):
    def __init__(self, filename):
        self.fin = open(filename, 'r')
        self.unicode = unicode
    def __iter__(self):
        self.fin.seek(0)
        for line in self.fin:
            yield line.decode('utf-8').split()

class Trie(object):
    def __init__(self):
        self.root = {}

    def add(self, word):
        if not isinstance(word, basestring):
            return
        word = word.split()
        node = self.root
        for l in word:
            if l not in node:
                node[l] = {}
            node = node[l]

    def __contains__(self, word):
        node = self.root
        word = word.split()
        for l in word:
            if l in node:
                node = node[l]
            else:
                return False
        return True

class BOW(object):
    def __init__(self, ngram):
        self.cache_uni = None
        self.cache_ngram = None
        self.map = None
        self.ngram = ngram

    def raw_count(self, file):
        """
        batch
        """
        logger.info('Extracting features: raw_count')
        ## get dimensions
        nfeatures = len(self.map)
        nrows = 0
        fs = open(file, 'r')
        for _ in fs:
            nrows += 1
        ## get features
        fs = open(file, 'r')
        label = np.empty(nrows, dtype=np.int)
        cache, row_idx, col_idx = [], [0], []
        for idx, line in enumerate(fs):
            if idx%10000 == 0:
                logger.debug('Finish {} samples, {}%'.format(idx, int(float(idx)/nrows*100)))
            line = line.decode('utf-8').strip().split(',')
            l = int(line[0])
            line = line[1:]
            label[idx] = l
            foo = defaultdict(int)
            for sent in line:
                sent = sent.split()
                ### add all unigram
                for w in sent:
                    if w in self.map:
                        foo[self.map[w]] += 1
                ### add ngram frome long to short
                # version 1 (include substr)
                for i in range(len(sent)-1):
                    for j in range(i+1, min(len(sent), i + self.ngram)):
                        w = ' '.join(sent[i:j + 1])
                        if w in self.map:
                            foo[self.map[w]] += 1
                # # version 2 (don't include substr)
                # i = 0
                # while i < len(sent) - 1:
                #     for j in range(min(len(sent), i + self.ngram) - 1, i, -1):
                #         w = ' '.join(sent[i:j + 1])
                #         if w in self.map:
                #             foo[self.map[w]] += 1
                #             i = j
                #     i += 1
            col_idx += foo.keys()
            cache += foo.values()
            row_idx.append(len(cache))
        data = sps.csr_matrix((cache, col_idx, row_idx), shape=(nrows, nfeatures), dtype=np.int)
        return data, label

    @staticmethod
    def _prune(cache, thre):
        tbd = []
        for key in cache.iterkeys():
            if cache[key] < thre:
                tbd.append(key)
        for key in tbd:
            del cache[key]
        del tbd

    def prune(self, cache, thre, maxmem):
        firsttimmer = 1
        while len(cache) > 0.5 * maxmem:
            if not firsttimmer:
                thre += 1
            self._prune(cache, thre)
            firsttimmer = 0
        return thre

    def count(self, file, maxmem=(1000000, 10000000), dic=set(), min_count=10, per=3000):
        """
        Count the frequency of unigram and ngram (up to self.ngram)
        :param file:
        :param maxmem: maximal memory usage
        :param dic: a dictionary to purify the count
        :param min_count:
        :param per:
        :return:
        """
        fp = open(file, 'r')
        cache_ngram = defaultdict(int)
        cache_uni = defaultdict(int)
        total = [0] * self.ngram
        thre = [2] * 2
        # track memory usage
        h = hpy()
        for idx, line in enumerate(fp):
            line = line.decode('utf-8').strip().split(',')[1:]
            if idx % per == 0:
                logger.debug('Finish {} lines with {} unigrams@{} and {} ngrams@{}'.format(idx, len(cache_uni), thre[0], len(cache_ngram), thre[1]))
                # print h.heap()
            for sent in line:
                sent = sent.split()
                for w in sent:
                    cache_uni[w] += 1
                    total[0] += 1
                i = 0
                while i < len(sent):
                    for j in range(min(len(sent), i + self.ngram) - 1, i, -1):
                        w = ' '.join(sent[i:j+1])
                        cache_ngram[w] += 1
                        total[j - i] += 1
                        if w in dic:
                            i = j
                            break
                    i += 1
                # prune
                if len(cache_uni) > maxmem[0]:
                    thre[0] = self.prune(cache_uni, thre[0], maxmem[0])
                if len(cache_ngram) > maxmem[1]:
                    thre[1] = self.prune(cache_ngram, thre[1], maxmem[1])
        # final prune:
        self._prune(cache_uni, max(min_count, thre[0]))
        self._prune(cache_ngram, max(min_count, thre[1]))
        return cache_uni, cache_ngram, total

    @staticmethod
    def bestN(cache_uni, cache_ngram, topN, mode='s'):
        if mode == 's':
            if isinstance(topN, int):
                cache_uni = sorted(cache_uni.items(), key=operator.itemgetter(1), reverse=True)[:topN]
                cache_ngram = sorted(cache_ngram.items(), key=operator.itemgetter(1), reverse=True)[:topN]
                return cache_uni, cache_ngram
            else:
                cache_uni = sorted(cache_uni.items(), key=operator.itemgetter(1), reverse=True)[:topN[0]]
                cache_ngram = sorted(cache_ngram.items(), key=operator.itemgetter(1), reverse=True)[:topN[1]]
                return cache_uni, cache_ngram
        elif mode == 'j':
            return sorted(cache_uni.items() + cache_ngram.items(), key=operator.itemgetter(1), reverse=True)[:topN]

    def save(self, filename):
        to_save = ['cache_uni', 'cache_ngram', 'map', 'ngram', 'cache_uni_wpmi']
        dic = self.__dict__
        tbs = {}
        for key in to_save:
            if key in dic:
                tbs[key] = dic[key]
        pickle.dump(tbs, open(filename, 'w'))

    def load(self, filename):
        tbd = pickle.load(open(filename, 'r'))
        for key, val in tbd.iteritems():
            self.__dict__[key] = val

class BOW_freq(BOW):
    def get_ngram(self, file, topN, maxmem=(1000000, 10000000), mode='s'):
        logger.info('BOW_freq: get_ngrams')
        cache_uni, cache_ngram, _ = self.count(file, maxmem=maxmem, min_count=10)
        #
        if mode == 's':
            cache_uni, cache_ngram = self.bestN(cache_uni, cache_ngram, topN, mode='s')
            foo = chain(cache_uni, cache_ngram)
        elif mode == 'j':
            foo = self.bestN(cache_uni, cache_ngram, topN, mode='j')
        self.map = {}
        for idx, w in enumerate(foo):
            w = w[0]
            self.map[w] = idx
        self.cache_uni = cache_uni
        self.cache_ngram = cache_ngram
        if mode == 's':
            logger.info('BOW_freq is finished; {} unigram features and {} ngram features'.format(len(cache_uni), len(cache_ngram)))
        elif mode == 'j':
            logger.info('BOW_freq is finished; {} unigram features and ngram features totally.'.format(len(self.map)))

class BOW_wpmi(BOW):
    @staticmethod
    def get_dic(cache):
        logger.info('BOW_wpmi: Generate Dictionary from current ngrams')
        dic = set()
        substr = Trie()
        for w, _ in cache:
            if w in substr:
                continue
            dic.add(w)
            w = w.split()
            for j in range(len(w) - 1):
                substr.add(' '.join(w[j:]))
        del substr
        logger.info('BOW_wpmi: Dictionary size {}'.format(len(dic)))
        return dic

    def get_map(self, cache_uni, cache_ngram):
        """
        Get ngram to index map
        :param cache_uni:
        :param cache_ngram:
        :return:
        """
        self.map = {}
        for idx, w in enumerate(chain(cache_uni, cache_ngram)):
            w = w[0]
            self.map[w] = idx


    def get_ngram(self, file, topN,
                  maxmem=(1000000, 10000000), niter=2, dic=set()):
        logger.info('BOW_wpmi: get_ngrams')
        for i in range(niter):
            # get counts
            cache_uni, cache_ngram, total = self.count(file, maxmem=maxmem, dic=dic, min_count=10)
            logtotal = [math.log(val) if val > 0 else None for val in total]
            # caclulate wpmi and store in cache_ngram
            min_val = None
            for val in cache_uni.itervalues():
                if not min_val or min_val > val:
                    min_val = val
            for key, val in cache_ngram.iteritems():
                words = key.split()
                l = len(words)
                logwords = [math.log(cache_uni[w]) if w in cache_uni else math.log(min_val) for w in words]
                cache_ngram[key] = val * (math.log(val) -
                                         logtotal[l - 1] -
                                         sum(logwords) +
                                         l * logtotal[0])
            # take best N
            cache_uni, cache_ngram = self.bestN(cache_uni, cache_ngram, topN)
            if i == niter - 1:
                break
            # get dic
            dic = self.get_dic(cache_ngram)
        # get ngram to index mapping
        self.get_map(cache_uni, cache_ngram)
        self.cache_uni, self.cache_ngram = cache_uni, cache_ngram
        logger.info('BOW_wpmi is finished; {} unigram features and {} ngram features are learned'.format(len(cache_uni), len(cache_ngram)))

    def rank_unigram_wpmi(self, file, maxmem=(1000000, 10000000)):
        # get counts
        self.ngram = 2
        cache_uni, cache_bi, total = self.count(file, maxmem=maxmem, min_count=10)
        logtotal = [math.log(val) if val > 0 else None for val in total]
        # get PMI for cache_bi
        for key, val in cache_bi.iteritems():
            words = key.split()
            if all([w in cache_uni for w in words]):
                cache_bi[key] = val * (math.log(val) -
                              logtotal[1] -
                              math.log(cache_uni[words[0]]) -
                              math.log(cache_uni[words[1]]) +
                              2 * logtotal[0])
        #
        cache_uni_wpmi = defaultdict(float)
        for key, val in cache_bi.iteritems():
            words = key.split()
            if all([w in cache_uni for w in words]):
                cache_uni_wpmi[words[0]] += val
                cache_uni_wpmi[words[1]] += val
        self.cache_uni, self.cache_uni_wpmi = cache_uni, cache_uni_wpmi


if __name__ == '__main__':
    text = smartfile('/media/vincent/Data-adhoc/wiki_dumps/wiki_zh/zhwiki-article')
    obj = BOW_wpmi(5)
    topN = 1000000
    obj.get_ngram(text, topN, niter=1)

    with open('zhwiki-article-{}gram-freq-list'.format(5), 'w') as h:
        for idx, item in enumerate(obj.cache_ngram):
            h.write('{}\t{}\t{}\n'.format(idx, item[0].encode('utf-8'), item[1]))

    dic = obj.get_dic(obj.cache_wpmi)
    obj.get_ngram(text, topN, niter=1, dic=dic)

    with open('zhwiki-article-{}gram-wpmi-list-iter{}'.format(5, 2), 'w') as h:
        for idx, item in enumerate(obj.cache_wpmi):
            h.write('{}\t{}\t{}\n'.format(idx,
                                          item[0].encode('utf-8'),
                                          math.log(item[1]) if item[1] > 0 else 0))







