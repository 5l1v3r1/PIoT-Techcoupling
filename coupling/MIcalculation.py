"""
encoding: UTF-8
author: W Q.Q.
组合不同领域并计算信息熵，按年计算指定耦合域互信息
"""
import pandas as pd
import numpy as np
import pickle
from itertools import product, combinations
from copy import deepcopy
import math
from tqdm import tqdm
from pprint import pprint

def comBine(cls, onlycombine=False):
    """
    返回cls中元素的二元、三元...的排列组合，onlycombine控制是否每个组合都要包含首元素
    :param onlycombine: True for combining all, False for cls[0]+comb(cls[0:])
    :return:
    """
    # cls = ['sa','ta','tb','tc']
    ele_num = len(cls)
    combine_list = []
    if onlycombine:
        for i in range(1, ele_num+1):
            tmp = [list(t) for t in combinations(cls, i)]
            combine_list.extend(tmp)
    else:
        if ele_num==1:
            return
        f_ele = cls[0]
        for i in range(1, ele_num):
            tmp = [list(t) for t in combinations(cls[1:], i)]
            combine_list.extend(tmp)
        for it in combine_list:
            if len(it)==1 and it[0]==f_ele:
                continue
            else:
                it.append(f_ele)
    return combine_list

def getDomainCount(combls, testnp, addmat):
    """get domains' term count by domain names"""
    sumcount = 0
    # b=[]
    for dns in combls:
        tmpindls = [domainID[d] for d in dns]
        if len(tmpindls) == 1:
            sumcount += sum(addmat[:, tmpindls[0]])
        else:
            tmp_subnp = testnp[:, tmpindls]
            for i in range(tmp_subnp.shape[0]):
                sumcount += min(tmp_subnp[i, :])
    return sumcount

def calcH(pls):
    """互信息公式"""
    h = sum(-p*math.log2(p) for p in pls if p!=0)
    return h

def getHX(X, domains, testnp, addmat, NC):
    """一维信息熵计算
    X:技术领域
    domains:8个技术子领域
    testnp:术语领域矩阵
    addmat:术语对领域矩阵
    NC：术语出现总次数
    """
    domain = deepcopy(domains)
    ind_x = domain.index(X)
    domain.remove(X)
    count_x = sum(testnp[:, ind_x])
    combls1 = comBine([X]+domain, onlycombine=False)
    print("HXcomb1_withX:", len(combls1), combls1)
    count_forp1 = getDomainCount(combls1, testnp, addmat)
    p1 = (count_x + count_forp1)/NC

    combls2 = comBine(domain, onlycombine=True)
    print("HXcomb2_noX:", len(combls2), combls2)
    p2 = getDomainCount(combls2, testnp, addmat)/NC
    return calcH([p1, p2])

def getHXY(X, Y, domains, testnp, addmat, NC):
    """二维信息熵"""
    # NC = sum(sum(testnp))
    domain1 = deepcopy(domains)# with X, not Y
    domain1.remove(X)
    domain1.remove(Y)
    combls1 = comBine([X]+domain1, onlycombine=False)
    print("HXYcomb1_withXnoY:", len(combls1), combls1)
    p1 = getDomainCount(combls1, testnp, addmat)/NC

    combls2 = comBine([Y]+domain1, onlycombine=False)   # with Y, not X
    print("HXYcomb2_withYnoX:", len(combls2),combls2)
    p2 = getDomainCount(combls2, testnp, addmat)/NC

    subcombls3 = comBine(domain1, onlycombine=True)   # NO X,Y
    print("HXYcomb3_noXnoY:", len(subcombls3),subcombls3)
    combls3 = [list(set([X, Y]+comb)) for comb in subcombls3]
    print("HXYcomb4_XandY:", len(combls3), combls3)
    p4 = getDomainCount(combls3, testnp, addmat)/NC
    p3 = getDomainCount(subcombls3, testnp, addmat)/NC
    return calcH([p1, p2, p3, p4])

def getMI(testnp, addmat):
    """get mutual information of each combs of the year"""
    MIls = []
    for ctup in couples:
        # newdomain = [ctup[0]] + iot_doms
        allcomb = comBine(domains, onlycombine=True)
        NC = getDomainCount(allcomb, testnp, addmat)

        # NC = sum(sum(addmat))
        print('testtup:', ctup)
        HX = getHX(ctup[0], domains, testnp, addmat, NC)
        print('HX:',HX)
        HY = getHX(ctup[1], domains, testnp, addmat, NC)
        print("HY:",HY)
        HXY = getHXY(ctup[0], ctup[1], domains, testnp, addmat, NC)
        print("HXY:", HXY)
        MIls.append(HX+HY-HXY)
        print("II:",HX+HY-HXY)
    return MIls


if __name__ == '__main__':

    path = r"D:\NLPcoupling\dataset\term_domDF_byyear1.pkl"
    with open(path, 'rb') as fp:
        termdomainls = pickle.load(fp)
    path1 = r"D:\NLPcoupling\dataset\termtup_domDF_byyear1.pkl"
    with open(path1, 'rb') as fp1:
        termtupdomls = pickle.load(fp1)
    eps_doms = ['EPS-Source', 'EPS-Grid', 'EPS-Load', 'EPS-Store']
    iot_doms = ['IOT-Percept', 'IOT-Network', 'IOT-Compute', 'IOT-App']
    domains = eps_doms+iot_doms
    domainID = dict(zip(eps_doms+iot_doms, list(range(8))))
    # couples = []    # coupling tuples of EPS and IoT
    # for ed in eps_doms:
    #     for id in iot_doms:
    #         couples.append((ed, id))
    couples = list(combinations(domains, 2))
    MIlist = []
    for i,termdomaindf in enumerate(termdomainls):
        testnp = termdomaindf.values
        matforsigTerm = termtupdomls[i][domains].values
        # testnpp = np.array([[0,2,1,2],[1,2,0,1],[0,0,1,4]])
        # allcomb = comBine(['d1','d2','d3','d4'], onlycombine=True)
        # domainID = dict(zip(['d1','d2','d3','d4'], list(range(4))))
        MIlist.append(getMI(testnp, matforsigTerm))

    df = pd.DataFrame(MIlist, index=list(range(2010, 2020)), columns=couples)
    df.to_excel(r"D:\NLPcoupling\dataset\termsRES\MI_SPECDOMAIN_NC_thred2_allcomb.xlsx")

    col = ["S", "G", "D", "E", "P", "N", "C", "A"]
    dom_abbr = dict(zip(domainID,col))
    tupcs = list(combinations(domains,2))
    colns = [dom_abbr[tup[0]]+"-"+dom_abbr[tup[1]] for tup in tupcs]

    ls_tups = []
    for i in range(10):
        tnp = termdomainls[i][domains].values
        adnp = termtupdomls[i][domains].values
        tmp = []
        for tup in tupcs:
            count = getDomainCount([tup],tnp,adnp)
            tmp.append(count)
        ls_tups.append(tmp)
    df_coups = pd.DataFrame(ls_tups,index=list(range(2010, 2020)), columns=colns)
    df_coups.to_excel(r"D:\NLPcoupling\dataset\forCase\term-coups0.xlsx")


    ls_term, ls_termtup = [],[]
    for i in range(10):
        mat = termdomainls[i][domains].values
        ls_term.append(sum(mat))
    df_term = pd.DataFrame(data=ls_term, index=list(range(2010, 2020)), columns=col)
    df_termtup = pd.DataFrame(data=ls_termtup, index=list(range(2010, 2020)), columns=col)
    df_term.to_excel(r"D:\NLPcoupling\dataset\forCase\term-d1.xlsx")
    df_termtup.to_excel(r"D:\NLPcoupling\dataset\forCase\termtup-d.xlsx")