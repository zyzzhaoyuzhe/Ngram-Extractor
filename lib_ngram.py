import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import numpy as np
from collections import defaultdict
import pickle
import operator
from itertools import chain
from copy import deepcopy

logger = logging.getLogger()
logger.setLevel(20)

class smartfile(object):
    def __init__(self, filename):
        self.fin = open(filename, 'r')
        # self.unicode = unicode

    def __iter__(self):
        self.fin.seek(0)
        for line in self.fin:
            yield line.split()


class Trie(object):
    def __init__(self):
        self.root = {}

    def add(self, word):
        if not isinstance(word, str):
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
        self._counts_ngrams = None  # list of dictionaries (ngram, freq)
        self.score_ngrams = None    # list of dictionaries (ngram, score)
        self.map = None             # dictionary (ngram, index)
        self.ngram = ngram          # int

    # def raw_count(self, file):
    #     """
    #     batch
    #     """
    #     logger.info('Extracting features: raw_count')
    #     ## get dimensions
    #     nfeatures = len(self.map)
    #     nrows = 0
    #     fs = open(file, 'r')
    #     for _ in fs:
    #         nrows += 1
    #     ## get features
    #     fs = open(file, 'r')
    #     label = np.empty(nrows, dtype=np.int)
    #     cache, row_idx, col_idx = [], [0], []
    #     for idx, line in enumerate(fs):
    #         if idx % 10000 == 0:
    #             logger.debug('Finish {} samples, {}%'.format(idx, int(float(idx) / nrows * 100)))
    #         line = line.decode('utf-8').strip().split(',')
    #         l = int(line[0])
    #         line = line[1:]
    #         label[idx] = l
    #         foo = defaultdict(int)
    #         for sent in line:
    #             sent = sent.split()
    #             ### add all unigram
    #             for w in sent:
    #                 if w in self.map:
    #                     foo[self.map[w]] += 1
    #             ### add ngram frome long to short
    #             # version 1 (include substr)
    #             for i in range(len(sent) - 1):
    #                 for j in range(i + 1, min(len(sent), i + self.ngram)):
    #                     w = ' '.join(sent[i:j + 1])
    #                     if w in self.map:
    #                         foo[self.map[w]] += 1
    #             # # version 2 (don't include substr)
    #             # i = 0
    #             # while i < len(sent) - 1:
    #             #     for j in range(min(len(sent), i + self.ngram) - 1, i, -1):
    #             #         w = ' '.join(sent[i:j + 1])
    #             #         if w in self.map:
    #             #             foo[self.map[w]] += 1
    #             #             i = j
    #             #     i += 1
    #         col_idx += foo.keys()
    #         cache += foo.values()
    #         row_idx.append(len(cache))
    #     data = sps.csr_matrix((cache, col_idx, row_idx), shape=(nrows, nfeatures), dtype=np.int)
    #     return data, label

    @staticmethod
    def _prune(cache, thre):
        tbd = []
        for key in cache.keys():
            if cache[key] < thre:
                tbd.append(key)
        for key in tbd:
            del cache[key]
        del tbd

    def prune(self, cache, thre, maxmem):
        while len(cache) > 0.5 * maxmem:
            self._prune(cache, thre)
            thre += 1
        return thre - 1

    def count(self, file, maxmem, dic=None, min_count=10):
        "Count the frequency of unigram and ngram (up to self.ngram)"
        h = open(file, 'r') if isinstance(file, str) else file

        cache_ngrams = [defaultdict(int) for _ in range(self.ngram)]
        total_ngrams = [0] * self.ngram

        thres = [2] * self.ngram
        for idx, line in enumerate(h):
            # line = line.decode('utf-8').strip().split(',')[1:]  # for Zhang's datasets
            line = [' '.join(line)]   # for wiki dumps
            if idx % 3000 == 0:
                logger.debug('Finish {} lines with {} n-grams and {} threshold'.format(idx, [len(cache) for cache in cache_ngrams], thres))

            for sent in line:
                sent = sent.split()
                len_sent = len(sent)

                sent_ngrams = [[' '.join(sent[left:left+k+1]) for k in range(self.ngram) if left+k+1 < len_sent] for left in range(len_sent)]

                for ngrams in sent_ngrams:
                    if not dic:
                        for k, w in enumerate(ngrams):
                            total_ngrams[k] += 1
                            cache_ngrams[k][w] += 1
                    else:
                        max_wpmi = None
                        for k, w in enumerate(ngrams[::-1]):
                            k = len(ngrams) - 1 - k
                            if max_wpmi is None or dic[k][w] > max_wpmi:
                                total_ngrams[k] += 1
                                cache_ngrams[k][w] += 1
                                max_wpmi = dic[k][w] if max_wpmi is None else max(max_wpmi, dic[k][w])

                # prune
                for k in range(self.ngram):
                    if len(cache_ngrams[k]) > maxmem[k]:
                        thres[k] = self.prune(cache_ngrams[k], thres[k], maxmem[k])

        # final prune:
        for k in range(self.ngram):
            if len(cache_ngrams[k]) > maxmem[k]:
                self._prune(cache_ngrams[k], thres[k], maxmem[k])

        self._counts_ngrams, self.total = cache_ngrams, total_ngrams

    def top_ngrams(self, topN):
        output = []
        for k in range(self.ngram):
            output.append(sorted(self.score_ngrams[k].items(), key=operator.itemgetter(1), reverse=True)[:topN[k]])
        return output

    def get_map(self):
        "Get ngram to index map"
        self.map = {}
        for idx, w in enumerate(chain(self._counts_ngrams)):
            w = w[0]
            self.map[w] = idx

    def save(self, filename):
        to_save = ['cache_uni', 'cache_ngram', 'map', 'ngram', 'cache_uni_wpmi', 'total']
        dic = self.__dict__
        tbs = {}
        for key in to_save:
            if key in dic:
                tbs[key] = dic[key]
        pickle.dump(tbs, open(filename, 'w'))

    def load(self, filename):
        tbd = pickle.load(open(filename, 'r'))
        for key, val in tbd.items():
            self.__dict__[key] = val


class BOW_freq(BOW):
    def fit(self, file, maxmem):
        logger.info('BOW_freq: get_ngrams')
        self.count(file, maxmem)
        self.score_ngrams = deepcopy(self._counts_ngrams)


class BOW_wpmi(BOW):

    def _get_wpmi(self):
        """ Calculate wpmi: p(w1,w2,..,wk) log(p(w1,w2,..,wk) / p(w1) p(w2) ... p(wk))"""
        self.score_ngrams = [defaultdict(int) for _ in range(self.ngram)]

        logtotal = np.log(self.total)

        cache_uni = self._counts_ngrams[0]
        for k in range(1, self.ngram):
            for ngram, freq in self._counts_ngrams[k].items():
                words = ngram.split()
                log_singleton = np.log([max(cache_uni[w], freq) for w in words])
                self.score_ngrams[k][ngram] = freq * (np.log(freq) - logtotal[k] - np.sum(log_singleton) + (k + 1) * logtotal[0])

        self.score_ngrams[0] = deepcopy(self._counts_ngrams[0])

    def fit(self, file, maxmem):
        logger.info('BOW_wpmi: get_ngrams')
        # first scan
        self.count(file, maxmem)
        # calculate wpmi
        self._get_wpmi()
        # second scan
        self.count(file, maxmem, dic=self.score_ngrams)
        self._get_wpmi()

    # def rank_unigram_wpmi(self, file, maxmem=(1000000, 10000000)):
    #     # get counts
    #     self.ngram = 2
    #     cache_uni, cache_bi, total = self.count(file, maxmem=maxmem, min_count=10)
    #     # logtotal = [math.log(val) if val > 0 else None for val in total]
    #     # # get PMI for cache_bi
    #     # for key, val in cache_bi.items():
    #     #     words = key.split()
    #     #     if all([w in cache_uni for w in words]):
    #     #         # weighted pmi
    #     #         # cache_bi[key] = val * (math.log(val) -
    #     #         #               logtotal[1] -
    #     #         #               math.log(cache_uni[words[0]]) -
    #     #         #               math.log(cache_uni[words[1]]) +
    #     #         #               2 * logtotal[0])
    #     #         # PMI
    #     #         v = (math.log(val) -
    #     #              logtotal[1] -
    #     #              math.log(cache_uni[words[0]]) -
    #     #              math.log(cache_uni[words[1]]) +
    #     #              2 * logtotal[0])
    #     #         cache_bi[key] = v
    #     # # get new rank
    #     # cache_uni_wpmi = defaultdict(float)
    #     # for key, val in cache_bi.items():
    #     #     words = key.split()
    #     #     if all([w in cache_uni for w in words]):
    #     #         cache_uni_wpmi[words[0]] += val
    #     #         cache_uni_wpmi[words[1]] += val
    #     self.cache_uni, self.cache_ngram = cache_uni, cache_bi
    #     self.total = total


if __name__ == '__main__':
    text = smartfile('zhwiki-article-tiny')

    topN = [1000000] * 5
    maxmem = [1000000] * 5
    obj = BOW_wpmi(5)
    obj.fit(text, maxmem)
    print(obj.top_ngrams(topN)[2][:20])


    # with open('zhwiki-article-{}gram-freq-list'.format(5), 'w') as h:
    #     for idx, item in enumerate(obj.cache_ngram):
    #         h.write('{}\t{}\t{}\n'.format(idx, item[0].encode('utf-8'), item[1]))
    #
    # dic = obj.get_dic(obj.cache_wpmi)
    # obj.fit(text, topN, niter=1, dic=dic)
    #
    # with open('zhwiki-article-{}gram-wpmi-list-iter{}'.format(5, 2), 'w') as h:
    #     for idx, item in enumerate(obj.cache_wpmi):
    #         h.write('{}\t{}\t{}\n'.format(idx,
    #                                       item[0].encode('utf-8'),
    #                                       math.log(item[1]) if item[1] > 0 else 0))
