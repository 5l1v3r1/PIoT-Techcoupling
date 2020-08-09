"""
encoding: UTF-8
author: W Q.Q.
获取按年的术语对-领域矩阵、术语-领域矩阵，以及全部数据集的总体矩阵
"""
import re, os
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
from tqdm import tqdm
import pickle, json
from pprint import pprint
import time
from preprocessing.dataConstruction import dataConstruction
import multiprocessing as mp
from itertools import combinations

class couplingMatrix():
    def __init__(self):
        fn = r'D:\NLPcoupling\dataset\resultsCoupling_onlyENTS2.pkl'
        cResults_for_all, cResults_by_year = self.readphraseData(fn)

        self.cpsDomains = ['EPS-Source', 'EPS-Grid', 'EPS-Load', 'EPS-Store']
        self.iotDomains = ['IOT-Percept', 'IOT-Network', 'IOT-Compute', 'IOT-App']

        self.ttup_domain_byear, self.term_domain_byear = self.getCouplingDF(cResults_by_year)
        self.ttup_domain_all, self.term_domain_all = self.getCouplingDF(cResults_for_all)

    def readphraseData(self, path):
        with open(path, 'rb') as fp:
            testCouplingRes = pickle.load(fp)
        all_year_counter, all_year_domain = [],[]
        for it in testCouplingRes:
            all_year_counter.extend(it['coupling_pairs'])
            all_year_domain.extend(it['domains'])
        all_couplingRes = [{'year':2000, 'coupling_pairs':all_year_counter, 'domains':all_year_domain}]
        # all_term_dfls, all_domain_dfls = getTermMatrix_total(all_couplingRes)
        # all_term_dfls[0].to_csv('D:\\NLPcoupling\\dataset\\termsRES\\all10.csv', sep=',',
        #                 encoding='utf-8', index=False)
        return all_couplingRes, testCouplingRes

    def mergePairs(self, ziptups, threshold=0):
        """将从dataConstruction中获取的文献实体及其出现次数数据转换为三元组，即实体对，频率，所属领域"""
        # cp_domain = zip(cps, dms)
        # ziptups = cp_domain
        longtriples_raw = []  # map pairs, domains, counts into a long list
        for pair, domain in ziptups:
            for tup, freq in pair.items():
                tmp = (tup, domain, freq)
                longtriples_raw.append(tmp)    # triples
        longtriples = []
        for trip in longtriples_raw:
            if not isinstance(trip[0][0], str) or not isinstance(trip[0][1], str):
                continue
            elif not trip[0][0] or not trip[0][1]:
                continue
            elif re.search(r'[:=]', trip[0][0]) or re.search(r'[:=]', trip[0][1]):
                continue
            elif self.isthesame(trip[0][0], trip[0][1]):
                continue
            else:
                longtriples.append(trip)
        print('longtriples len{}'.format(len(longtriples)))
        #remove duplicates
        shortrips = []
        for i in tqdm(range(len(longtriples))):
            tmp = list(longtriples[i])
            tripls = [trip[0] for trip in shortrips]
            if tmp[0] in tripls or (tmp[0][1], tmp[0][0]) in tripls:
                continue
            else:
                for j in range(i + 1, len(longtriples)):
                    trpj = longtriples[j]
                    if trpj[0] == tmp[0] or (trpj[0][1], trpj[0][0]) == tmp[0]:
                        if trpj[1] == tmp[1]:
                            tmp[2] += trpj[2]
                        else:
                            continue
                    else:
                        continue
                shortrips.append(tmp)
        print('shortriples len{}'.format(len(shortrips)))
        filtered_trips = []
        for trip in shortrips:
            if trip[2] > threshold:
                tmp = tuple(trip)
                filtered_trips.append(tmp)
        return filtered_trips

    def getCouplingDF(self, cresults):
        """术语对-领域矩阵和术语-领域矩阵获取"""
        dfls_coupDom, dfls_coupTerm = [],[]
        for item in tqdm(cresults):
            year = item['year']
            coupling_pairs = item['coupling_pairs']
            domains = item['domains']
            df, df1 = self.gettupdMatrix(year, coupling_pairs, domains)
            dfls_coupDom.append(df)
            dfls_coupTerm.append(df1)
        return dfls_coupDom, dfls_coupTerm

    def gettupdMatrix(self, yr, cpls, dmls):
        print('Processing data of year {}'.format(yr))
        cps, dms = cpls.copy(), dmls.copy()
        for i, c in enumerate(cpls):
            if not c:
                cps.remove(c)
                dms.remove(dmls[i])
        print('Valid papers are {}'.format(len(cps)))
        cp_domain = zip(cps, dms)

        term_domain = Counter()  # term in domain Count
        termtuple_domain = Counter()
        source_terms, target_terms, pcountls = [], [], []  # term set

        print('Convert all data to non-duplicating Triples>>>')
        pair_domain_ctrips = self.mergePairs(cp_domain, threshold=2)  # get all paper counter together, remove duplications
        for trip in pair_domain_ctrips:
            term_domain[(trip[0][0], trip[1])] += trip[2]  # the count of the term occurs in a domain
            term_domain[(trip[0][1], trip[1])] += trip[2]
            termtuple_domain[((trip[0][0],trip[0][1]),trip[1])] += trip[2]
            source_terms.append(trip[0][0])
            target_terms.append(trip[0][1])
            pcountls.append(trip[2])
        termls = list(set(source_terms + target_terms)) # get all terms
        print('Get {} terms.'.format(len(termls)))
        termtupls = list(set([tup[0] for tup in termtuple_domain.keys()]))
        domain_names = self.cpsDomains + self.iotDomains
        ttupdomain_matrix = np.array(np.zeros((len(termtupls), 8)))
        for ttupd, count in termtuple_domain.items():
            rowind = termtupls.index(ttupd[0])
            colind = domain_names.index(ttupd[1])
            ttupdomain_matrix[rowind][colind] = count
        df = pd.DataFrame(ttupdomain_matrix, columns=domain_names)
        df['SN'] = [tup[0] for tup in termtupls]
        df['TN'] = [tup[1] for tup in termtupls]

        termdomain_matrix = np.array(np.zeros((len(termls), 8)))
        for tupd, count in term_domain.items():
            rowind = termls.index(tupd[0])
            colind = domain_names.index(tupd[1])
            termdomain_matrix[rowind][colind] = count
        df1 = pd.DataFrame(termdomain_matrix,columns=domain_names)
        df1['T'] = termls
        return df, df1

    def isthesame(self,a,b):
        lsa = a.lower().split(' ')
        lsb = b.lower().split(' ')
        if len(lsa) < 2 and len(lsb) < 2:
            return False
        else:
            if lsa == lsb:
                return True
            else:
                return False

    def saveTopick(self, ob, spath):
        with open(spath, 'wb') as fp:
            pickle.dump(ob, fp)
