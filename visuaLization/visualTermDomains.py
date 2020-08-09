from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import pickle

rootpath = r"D:\NLPcoupling\dataset\forCytoscape"
fns = [os.path.join(rootpath, f) for f in os.listdir(rootpath) if "default node" in f]

dfls = []
for fn in fns:
    df = pd.read_csv(fn)
    sortbyBC = df.sort_values(["BetweennessCentrality"], ascending=False)
    subdf = sortbyBC[["BetweennessCentrality", "COUNT", "DOMS", "name"]].iloc[0:50, :]
    dfls.append(subdf)

termls = defaultdict(list)
for i, tmpdf in enumerate(dfls):
    for tn in tmpdf["name"].to_list():
        termls[tn] = [0]*10
for i, tmpdf in enumerate(dfls):
    for j,tn in enumerate(tmpdf["name"].to_list()):
        termls[tn][i] = tmpdf["BetweennessCentrality"].values[j]


# satistermls = defaultdict(list)
# for k,v in termls.items():
#     if len(v)==10:
#         satistermls[k] = v

critical_tdf = pd.DataFrame(termls)
critical_tdf["year"] = list(set(range(2010, 2020)))
critical_tdf.to_excel(r"D:\NLPcoupling\dataset\MIIIII\criticalterms.xlsx")


termpath = r"D:\NLPcoupling\dataset\term_domDF_byyear1.pkl"
with open(termpath,'rb') as fp:
    termdfls = pickle.load(fp)

ctermdic = defaultdict(list)
for k,v in termls.items():
    ctermdic[k] = [0]*10
ctermdic["ECG"] = [0]*10
for i, termdf in enumerate(termdfls):
    termmat = termdf[['EPS-Source', 'EPS-Grid', 'EPS-Load', 'EPS-Store', 'IOT-Percept',
       'IOT-Network', 'IOT-Compute', 'IOT-App']].values

    for term in ["ECG"]:
        for j, target_t in enumerate(termdf["T"].to_list()):
            if term == target_t:
                ctermdic[term][i] = sum(termmat[j,:])
c_tdf = pd.DataFrame(ctermdic)
c_tdf["year"] = list(set(range(2010, 2020)))
c_tdf.to_excel(r"D:\NLPcoupling\dataset\MIIIII\ctermssssss1.xlsx")