import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import numpy as np
import scipy.sparse as sps
from collections import defaultdict
from nltk.tokenize import word_tokenize
import operator
from itertools import chain, tee
import math
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
    def __init__(self):
        self.cache_uni = None
        self.cache_ngram = None
        self.map = None
        self.ngram = None

    def raw_count(self, text, maxmem=1500000000):
        """
        batch
        """
        logger.debug('Extracting features: raw_count')
        text, text_copy = tee(text)
        ## get dimensions
        nfeatures = len(self.map)
        # nrows = int(float(maxmem) / nfeatures / np.dtype(float).itemsize)
        nrows = 0
        for _ in text:
            nrows += 1
        ## get features
        # logger.info("estimated required memory {} bytes".format(nrows * nfeatures * np.dtype(float).itemsize))
        label = np.empty(nrows, dtype=np.int)
        cache, row_idx, col_idx = [], [0], []
        for idx, line in enumerate(text_copy):
            if idx%10000 == 0:
                logger.debug('Finish {} lines, {}%'.format(idx, int(float(idx)/nrows*100)))
            l, line = line
            label[idx] = l
            foo = defaultdict(int)
            for sent in line:
                # add all unigram
                for w in sent:
                    if w in self.map:
                        foo[self.map[w]] += 1
                # add ngram frome long to short
                # version 1
                for i in range(len(sent)-1):
                    for j in range(i+1, min(len(sent), i + self.ngram)):
                        w = ' '.join(sent[i:j + 1])
                        if w in self.map:
                            foo[self.map[w]] += 1
                # # version 2
                # i = 0
                # while i < len(sent) - 1:
                #     for j in range(min(len(sent), i + self.ngram) - 1, i, -1):
                #         w = ' '.join(sent[i:j + 1])
                #         if w in self.map:
                #             foo[self.map[w]] += 1
                #             i = j
                #     i += 1
            foo = foo.items()
            col_idx += [w[0] for w in foo]
            cache += [w[1] for w in foo]
            row_idx.append(len(cache))
        data = sps.csr_matrix((cache, col_idx, row_idx), shape=(nrows, nfeatures), dtype=np.int)
        return data, label

    @staticmethod
    def _prune(cache, thre):
        for key in list(cache):
            if cache[key] < thre:
                del cache[key]

    def prune(self, cache, thre, maxmem):
        firsttimmer = 1
        while len(cache) > 0.5 * maxmem:
            if not firsttimmer:
                thre += 1
            self._prune(cache, thre)
            firsttimmer = 0
        return thre

    def count(self, text, maxmem=10000000, dic=set(), min_count=10, per=10000):
        cache_ngram = defaultdict(int)
        cache_uni = defaultdict(int)
        total = [0] * self.ngram
        thre = [2] * 2
        for idx, sent in enumerate(text):
            if idx % per == 0:
                logger.debug('Finish {} lines with {} unigrams and {} ngrams'.format(idx, len(cache_uni), len(cache_ngram)))
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
            if len(cache_ngram) > maxmem:
                thre[1] = self.prune(cache_ngram, thre[1], maxmem)
            if len(cache_uni) > maxmem:
                thre[0] = self.prune(cache_uni, thre[0], maxmem)

        # final prune:
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


class BOW_freq(BOW):
    def __init__(self, ngram):
        super(BOW_freq, self).__init__()
        self.ngram = ngram

    def get_ngram(self, text, topN, maxmem=10000000, mode='s'):
        logger.info('BOW_freq: get_ngrams')
        cache_uni, cache_ngram, _ = self.count(text, maxmem=maxmem, min_count=10)
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
            logger.info('{} unigram features and {} ngram features'.format(len(cache_uni), len(cache_ngram)))
        elif mode == 'j':
            logger.info('{} unigram features and ngram features totally.'.format(len(self.map)))

class BOW_wpmi(BOW):
    def __init__(self, ngram):
        super(BOW, self).__init__()
        self.ngram = ngram
        self.cache_wpmi = None

    @staticmethod
    def get_dic(cache):
        dic = set()
        substr = Trie()
        for w, _ in cache:
            if w in substr:
                continue
            dic.add(w)
            w = w.split()
            for j in range(len(w) - 1):
                substr.add(' '.join(w[j:]))
        return dic

    def get_ngram(self, text, topN, maxmem=10000000, niter=2, dic=set()):
        logger.info('BOW_wpmi: get_ngrams')
        text = tee(text, niter)
        for i in range(niter):
            logger.info('Dictionary size {}'.format(len(dic)))
            cache_uni, cache_ngram, total = self.count(text[i], maxmem=maxmem, dic=dic, min_count=10)
            logtotal = [math.log(val) if val > 0 else None for val in total]
            # caclulate wpmi
            cache_wpmi = {}
            for key, val in cache_ngram.iteritems():
                words = key.split()
                l = len(words)
                cache_wpmi[key] = val * (math.log(val) -
                                         logtotal[l - 1] -
                                         sum(math.log(cache_uni[w]) for w in words) +
                                         l * logtotal[0])
            # take best N
            cache_uni, cache_wpmi = self.bestN(cache_uni, cache_wpmi, topN)
            if i == niter - 1:
                break
            # get dic
            dic = self.get_dic(cache_wpmi)
        # get ngram to index mapping
        self.map = {}
        for idx, w in enumerate(chain(cache_uni, cache_wpmi)):
            w = w[0]
            self.map[w] = idx
        self.cache_uni = cache_uni
        _, cache_ngram = self.bestN({}, cache_ngram, topN)
        self.cache_ngram = cache_ngram
        self.cache_wpmi = cache_wpmi
        logger.info('{} unigram features and {} ngram features'.format(len(cache_uni), len(cache_wpmi)))



if __name__ == '__main__':
    text = smartfile('/media/vincent/Data-adhoc/wiki_dumps/wiki_zh/zhwiki-article')
    obj = BOW_wpmi(5)
    topN = 10000
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







