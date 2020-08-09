"""
using NLTK to preprocess WOS raw title,abstract,keywords
including: word tokenize; lemmatize; pos tagging
target format: [{tit_kws:[sentwords],abs:[[sentwords]]},{},...]
"""
import pandas as pd
import spacy, nltk
import re, os
from tqdm import tqdm
from collections import Counter
# from collections import defaultdict
import spacy.lang.en.stop_words as stw
import Levenshtein as lvst
from itertools import combinations
import random

class dataConstruction():
    def __init__(self):
        rawcsvpath = 'D:\\NLPcoupling\\dataset\\raw_wos.csv'
        structuredpath = 'D:\\NLPcoupling\\dataset\\structured_data.csv'
        wos_df = pd.read_csv(rawcsvpath)

        self.abbrev_dict = {}   # initialize abbreviation dict
        # if os.path.exists(structuredpath):
        #     self.structured_data = pd.read_csv(structuredpath)
        # else:
        self.structured_data = self.data_initialize(wos_df, savedata=True)    # processed data

        self.critical_terms = self.generateSeeds()  # create golden seeds
        self.corefls = self.Corefwords()


    def data_initialize(self, whole_wosdata, savedata=False):
        """extract keywords as gold seeds, reorganize rawdata"""
        structured_data = whole_wosdata.copy()
        kwChunker = re.compile(r'[;,\n]')   # for keywords chunking
        kwls = []   # list of keywords list, and a dict of abbreviations
        print('Initializing seed words:')
        for kwords in tqdm(structured_data['KW']):
            if type(kwords) == str:
                kws = [w.lstrip(' ') for w in re.split(kwChunker, kwords) if w]
                templs = []
                for kw in kws:
                    # templs.extend([w.strip(')').rstrip(' ') for w in kw.split('(')])
                    pat_abbrev = re.compile(r'\(([\w-]*[A-Z]+[\w-]*)\)')  # match abbreviation
                    abbmatch = [w for w in re.findall(pat_abbrev, kw) if
                                len(w) > 1]  # find all abbreviations in the chunk,filter sigle letter
                    if not abbmatch:
                        templs.append(kw)
                    else:
                        templs.extend(abbmatch)
                        if abbmatch[0] not in self.abbrev_dict.keys():
                            temp = kw.split('(')[0]
                            if temp:
                                self.abbrev_dict.update({abbmatch[0]: temp.rstrip(' ')})
                        clean_bracket = re.sub(pat_abbrev, '', kw)
                        clean_kw = re.sub(r'[ ]{2,}', '', clean_bracket).rstrip(' ').lstrip(' ')
                        templs.append(clean_kw)
                kwls.append(templs)
            else:
                kwls.append([])
        print("Reorganizing data structure:")
        merged_col, merged_col1 = [],[]  # merge title and abstract to a sentence list
        for i in tqdm(range(structured_data.shape[0])):
            abs_sentls = nltk.sent_tokenize(structured_data['AB'].iloc[i])
            normalized_tit = self.normalizeSent(structured_data['TI'].iloc[i])
            abs_sentls.insert(0, normalized_tit)
            merged_col.append(abs_sentls)
            abs_sentclean = []
            for sent in abs_sentls:
                csent = self.cleanSent(sent)
                if csent:
                    abs_sentclean.append(csent)
                self.getAbbrevs(sent)
            merged_col1.append(abs_sentclean)
        structured_data.loc[:, "KW"] = kwls  # replace original KW column with keywords list
        structured_data.loc[:, "TI_AB"] = merged_col     # create a new col of merged title and abstract
        structured_data.loc[:, "TI_AB_CLEAN"] = merged_col1  # create a new col of merged title and abstract
        if savedata:
            structured_data.to_csv('D:\\NLPcoupling\\dataset\\structured_data.csv', sep=',',
                                   encoding='utf-8', index=False)
        return structured_data

    def phrase_extracting(self, threshold_sent = 1, threshold_pap = 0):
        """
        using trained Spacy model to extract phrases in each sentence
        threshold_sent    threshold_pap
            >0                0           focus on the coherence in a sentence
            0                >0           focus on the coherence in a title+abstract text
        """
        # load customized NER model
        nlp = spacy.load('D:/NLPcoupling/nlpmodels/nlp0.2')
        # for each year and each tit_ab sents of that year, extract phrases to T1 and T2
        # coocur_overtime = {}
        termCouplingResults = []
        print('Constructing tech term coupling matrix:')
        for i in range(2010, 2020):
            # coocur_overtime['year'] = i
            df_by_year = self.structured_data[self.structured_data['YEAR'] == i]
            # df_by_year = datacons.structured_data[datacons.structured_data['YEAR'] == 2010]
            couplingTerm_dict = {}      # {'year':'','coupling_pairs':[],'domains':[]}
            coupling_pairs, domains = [], []
            print("processing year {}, with {} rows>>>".format(i, df_by_year.shape[0]))
            for row in tqdm(range(df_by_year.shape[0])):
                pair_for_pap = Counter()    # term pair counter in the paper tit_ab
                tempdf = df_by_year.iloc[row, :]   # get row
                kwords, sentChunk, domain = tempdf[['KW', 'TI_AB_CLEAN', 'DOMAIN']]  # get row data
                # neat_kwords = self.resolutingList(kwords)   # some papers may get no keyword
                for sent in sentChunk:
                    pair_for_sent = Counter()
                    doc = nlp(sent)
                    ents = [e.text for e in doc.ents if e.label_ == 'TECH']
                    bool_exit = len(ents) < 2 and not kwords
                    if not ents or bool_exit:
                        continue
                    # nouns = self.nounChunk_clean(list(doc.noun_chunks))   # 测试
                    combls = self.combineLists(kwords, ents)
                    # combls = combinations(allents,2)  # 测试
                    for tup in combls:
                        pair_for_sent[tup] += 1
                    for tup, c in pair_for_sent.items():
                        if c > threshold_sent:
                            pair_for_pap.update({tup: c})
                merged_pair_pap = self.mergePairs(pair_for_pap)
                filtered_pap = Counter()
                for tup, c in merged_pair_pap.items():
                    if c > threshold_pap:
                        filtered_pap.update({tup: c})  # get term pairs of each paper, for further usage
                if filtered_pap:
                    coupling_pairs.append(filtered_pap)
                    domains.append(domain)
                else:
                    coupling_pairs.append(Counter())
                    domains.append(domain)
            couplingTerm_dict.update({'year': i, 'coupling_pairs': coupling_pairs, 'domains': domains})
            termCouplingResults.append(couplingTerm_dict)
        return termCouplingResults

    def mergePairs(self, pair):
        """merge two pairs with same tuples, pair: Class Counter"""
        newpair = Counter()
        for tup, count in pair.items():
            if tup in newpair.keys():
                newpair[tup] += count
            elif (tup[1], tup[0]) in newpair.keys():
                newpair[(tup[1], tup[0])] += count
            else:
                newpair[tup] = count
        return newpair

    def combineLists(self, kws, ents):
        """combine elements in two lists, filter duplications"""
        tupls = []
        if kws and ents:
            for it in combinations(kws + ents, 2):
                juj1 = bool(it[0] in kws and it[1] in kws)
                juj2 = bool(it[0] in ents and it[1] in ents)
                juj3 = bool(it[0] != it[1])
                if not self.isconref(it) and not juj1 and not juj2 and juj3:
                    tupls.append(it)
        else:
            return list(combinations(ents, 2))
        return tupls

    def isconref(self, tup):
        for it in self.corefls:
            if tup[0] in it and tup[1] in it:
                return True
        return False

    def resolutingList(self, ls):
        """coref resolution for a ls"""
        if not ls:
            return []
        newls = ls.copy()
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                temp = self.corefResolute(ls[i], ls[j])
                if temp:
                    if ls[i] in newls:
                        newls.remove(ls[i])
                    if ls[j] in newls:
                        newls.remove(ls[j])
                    newls.append(temp)
        return newls

    def corefResolute(self, t1, t2):
        """共指代消解"""
        isAbbr = lambda term: True if len([l for l in term if l.isupper()])>1 else False
        if isAbbr(t1) and not isAbbr(t2):
            if t1 in self.abbrev_dict.keys():
                if self.abbrev_dict[t1].lower() in t2.lower() or t2.lower() in self.abbrev_dict[t1].lower():
                    return t2
        elif isAbbr(t2) and not isAbbr(t1):
            if t2 in self.abbrev_dict.keys():
                if self.abbrev_dict[t2].lower() in t1.lower() or t1.lower() in self.abbrev_dict[t2].lower():
                    return t1
        elif isAbbr(t1) and isAbbr(t2):
            if t1.rstrip('s') == t2.rstrip('s'):
                if len(t1) > len(t2):
                    return t1
                else:
                    return t2
        else:
            if lvst.distance(t1.lower(), t2.lower()) < 1:
                return t1
        return ''

    def nounChunk_clean(self, noun_chunkls):
        """清洗名词术语"""
        newChunk = []
        stopwords = stw.STOP_WORDS
        for noun in noun_chunkls:
            nountext = noun.text
            if isinstance(nountext, str) and not nountext.endswith('ing'):
                templs = nltk.word_tokenize(nountext.lower())
                for token in templs:
                    if token in stopwords:
                        break
                    newChunk.append(nountext)
        return newChunk

    def normalizeSent(self, title):
        """convert termpair dict to coupling matrix, return term freq"""
        tokenls = title.split(' ')
        newtokens = []
        for token in tokenls:
            if not re.search('[A-Z]{2,}', token):
                newtokens.append(token.lower())
            else:
                newtokens.append(token)
        newtitle = ' '.join(newtokens).capitalize()
        return newtitle


    def getAbbrevs(self, chunk):
        """match abbreviations, if possible, get explanations"""
        pat_abbrev = re.compile(r'\(([\w-]*[A-Z]+[\w-]*)\)') # match abbreviation
        abbmatch = [w for w in re.findall(pat_abbrev, chunk) if
                    len(w) > 1]  # find all abbreviations in the chunk,filter sigle letter
        if not abbmatch:
            return
        else:
            item_ind_tail = 0   # tail index in the chunk
            for item in set(abbmatch):
                item_ind_head = chunk.index('('+item+')')   # get head index of abbrev
                tempstr = chunk[item_ind_tail:item_ind_head].lstrip(' ')
                templs = nltk.word_tokenize(tempstr)     # split text before the abbrev
                if len(templs) >= len(item):    # detect words before the abbrev
                    subls = templs[-len(item):]
                    if item not in self.abbrev_dict.keys():
                        if item.lower() == ''.join([w.lower()[0] for w in subls]):  # match acronym in the forehead words
                            self.abbrev_dict[item] = ' '.join(subls)
                        else:
                            return
                else:
                    return
                item_ind_tail = item_ind_head + len(item) + 2

    def cleanSent(self, sentence):  # check
        """simple preprocess"""
        csent = re.sub(r'(\(.*\))|[\n]', '', sentence)
        csent = re.sub(r' {2,}', ' ', csent).rstrip(' ')
        # add a tag to the end of each sentence
        if len(csent) < 3:
            return ''
        if re.match(r'[\W]', csent[-1]):
            csent = csent[:-1]
        return csent

    def generateSeeds(self):
        """随机选取关键词，以备抽取包含关键词的句子组成训练集"""
        abbrev_dic_items = list(self.abbrev_dict.items())
        itlength = len(abbrev_dic_items)
        random.shuffle(abbrev_dic_items)
        kw_seeds = [k for k, _ in abbrev_dic_items[:int(itlength/2)] if not re.search(r'[\(\)\-\[\]]', k)]
        vl_seeds = [v for _, v in abbrev_dic_items[int(itlength/2):] if not re.search(r'[\(\)\[\]]', v)]

        # kw_seeds = [(k, 'ABBR') for k, _ in abbrev_dic_items]
        # kw_seeds = [k for k, _ in abbrev_dic_items]
        # for kw, _ in kw_seeds.copy():  # extend abbreviations by detecting suffix
        #     if kw.endswith('s') and kw.rstrip('s') not in self.abbrev_dict.keys():
        #         kw_seeds.append((kw.rstrip('s'), 'ABBR'))
        # seed_phrases = kw_seeds + [(v.lower(), 'EXTN') for _, v in abbrev_dic_items]  # golden seed words
        seed_phrases = kw_seeds + vl_seeds  # golden seed words
        return seed_phrases

    def Corefwords(self):
        """具有相同意义的单词"""
        print("Complete conreference word list.")
        corefs = [[k,v] for k,v in self.abbrev_dict.items()]
        num = len(corefs)
        for i in tqdm(range(num)):
            for j in range(i+1, num):
                ci, cj = corefs[i], corefs[j]
                if self.isame(ci, cj):
                    ci.extend(cj)
        newcorefs = self.extendls(corefs)
        neat_refls = []
        for it in newcorefs:
            if it not in neat_refls:
                neat_refls.append(it)
        with open(r"D:\NLPcoupling\auxdata\corefwords.txt", 'w') as fp:
            for line in neat_refls:
                for ele in line:
                    fp.write(str(ele) + "\t")
                fp.write("\n")
        return neat_refls


    def isame(self,tupa,tupb):
        tupab = [(s,t) for s,t in zip(tupa,tupb)]
        tupb.reverse()
        tupab.extend([(s,t) for s,t in zip(tupa,tupb)])
        for a,b in tupab:
            lsa = a.lower().split(' ')
            lsb = b.lower().split(' ')
            if len(lsa) < 2 and len(lsb) < 2:
                if a==b:
                    return True
                else:
                    return False
            else:
                if lsa == lsb:
                    return True
                else:
                    return False

    def extendls(self, ls):
        print("Extending conref list.")
        diydic = {"Smart grid":["smart grid","Smart Grid","SG","SGs"],
                  "MGs":["microgrid","Microgrid","MG"],"PV":["Photovoltanic",'Photovoltanics']}
        newls = []
        for it in tqdm(ls):
            tmp=[]
            for kw in diydic.keys():
                if kw in it:
                    tmp.extend(diydic[kw])
                else:
                    continue
            for item in it:
                if len(item.split(' '))<2:
                    tmp.append(item)
                else:
                    lowerit = item.lower()
                    extmp = [item, lowerit, lowerit.capitalize()]
                    tmp.extend(extmp)
            newls.append(list(set(tmp)))
        return newls