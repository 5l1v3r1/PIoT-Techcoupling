"""筛选数据作图"""
import pickle
import numpy as np
import pandas as pd

path1 = r"D:\NLPcoupling\dataset\termtup_domDF_byyear.pkl"
with open(path1, 'rb') as fp1:
    term_tup_domainls = pickle.load(fp1)
path2 = r"D:\NLPcoupling\dataset\term_domDF_byyear.pkl"
with open(path2, 'rb') as fp2:
    term_domainls = pickle.load(fp2)

for i in range(9):
    fn1 = r"D:\NLPcoupling\dataset\termsRES\terms"+str(i)+".csv"
    df = pd.read_csv(fn1)

    newdf = df[df["Pair Count"]>2]
    newdf.to_excel(r"D:\NLPcoupling\dataset\termsRES\terms_"+str(i)+".xlsx")