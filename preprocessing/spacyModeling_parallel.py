"""
Customized class for PIoT phrase detection via SpaCy functions,
Training a customized NER model from seed keywords
"""
import re, os
import json

# Train NER from a blank spacy model
from pprint import pprint
import spacy
import spacy.gold
from tqdm import tqdm
import pandas as pd
from functools import partial
from spacy.util import minibatch, compounding, decaying
import random
import multiprocessing as mp
from preprocessing.dataConstruction import dataConstruction
import time
# print("Number of processors: ", mp.cpu_count())
import pickle


def trainingSet_generate(sentls, critical_terms):
    """generate training set for spaCy NER"""
    tmp_trainingset = []
    # critical_terms = datacons.critical_terms
    # print("Construct training set:")
    # training_data = []
    # rownums = wos_df.shape[0]
    # for i in tqdm(range(rownums)):
    # sentls = wos_df['TI_AB_CLEAN'][i]

    if len(sentls) > 1:
        sentls.remove(sentls[1])
    for sent in sentls:
        maxterms = 1    # max num of word detected in one single sentence
        subtrainingset = get_wordLoc(critical_terms, sent, maxTerm=maxterms)
        if not subtrainingset:
            continue
        else:
            tmp_trainingset.extend(subtrainingset)
    print("\t Data saturating...")
    return tmp_trainingset

def get_wordLoc(critical_terms, sentence, maxTerm):
    """get word location in the sentence(head ind, tail ind), each sent may gets more than 1 words inside"""
    label = 'TECH'  # new label for ner
    triplels = []
    goldterm = critical_terms
    random.shuffle(goldterm)
    for seed in goldterm:
        # pat1 = re.compile()
        try:
            pat = re.compile("(^{}[ ])|([ ]{}[ ])|([ ]{}$)".format(seed,seed,seed))
            match1 = re.search(pat, sentence)
            if match1:
                matstr = match1.group()
                matspan = match1.span()
                if len(matstr)-len(seed) == 2:
                    triplels.append((matspan[0]+1, matspan[1]-1, label))
                elif matstr.startswith(' '):
                    triplels.append((matspan[0] + 1, matspan[1], label))
                else:
                    triplels.append((matspan[0], matspan[1]-1, label))
            else:
                continue
        except:
            pass
        if len(triplels) >= maxTerm:
            subset = [(sentence, {"entities": [triple]}) for triple in triplels]  # piece(s) of gold sentence
            return subset
    if triplels:
        subset = [(sentence, {"entities": [triple]}) for triple in triplels]  # piece(s) of gold sentence
        return subset
    else:
        return []

        # pat2 = re.compile(r"[\W]{}[\W]".format(seed), re.I)
        # match2 = re.search(pat2, sentence)
        # if match2 and tag == 'EXTN':
        #     triplels.append((match2.span()[0]+1, match2.span()[1]-1, label))
        #     continue

    # if triplels:
    #     subset = [(sentence, {"entities": [triple]}) for triple in triplels]  # piece(s) of gold sentence
    #     return subset
    # return []

def nerModeling(training_data):
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe('ner'))
    nlp.begin_training()

    #load the spacy model
    nlp = spacy.load("en_core_web_sm")
    # Getting the ner component
    ner = nlp.get_pipe('ner')

    # Add the new label to ner
    ner.add_label('TECH')

    # Resume training
    optimizer = nlp.resume_training()
    move_names = list(ner.move_names)

    # List of pipes you want to train
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    # List of pipes which should remain unaffected in training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Begin training by disabling other pipeline components
    # dropout = decaying(0.6, 0.2, 1e-4)

    # Training for 30 iterations
    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 32.0, 1.001)
        for itn in range(30):
            # shuffle examples before training
            random.shuffle(training_data)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(training_data, size=sizes)
            # dictionary to store losses
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                # Calling update() over the iteration
                nlp.update(texts, annotations, sgd=optimizer, drop=0.3, losses=losses)
                print("Losses", losses)
    nlp.meta['name'] = 'testmodel'  # rename model
    nlp.to_disk(r'D:/NLPcoupling/nlpmodels/nlp0.2')
    print('Model saved.')

def mergeData():
    filename = "D:/NLPcoupling/dataset/trainingset_merged.json"
    with open(filename, 'r') as file_obj:
        data = json.load(file_obj)

    filename1 = "D:/NLPcoupling/dataset/trainingset1_0.2.pkl"
    with open(filename1, 'rb') as file_obj:
        data1 = pickle.load(file_obj)
    mergeddata = []
    for it1 in data:
        tmp = tuple(it1[1]['entities'][0])
        tmpdic = {"entities": [tmp]}
        if tmpdic not in mergeddata:
            mergeddata.append((it1[0], tmpdic))
    for it2 in data1:
        if it2 not in mergeddata:
            mergeddata.append(it2)
    filename2 = "D:/NLPcoupling/dataset/trainingset_merged.pkl"
    with open(filename2, 'wb') as file_obj:
        pickle.dump(mergeddata, file_obj)
    filename3 = "D:/NLPcoupling/dataset/trainingset_merged.json"
    with open(filename3, 'w') as file_obj:
        json.dump(mergeddata, file_obj)

def filter_data(rawdata):
    """check enetity alignment"""
    nlp = spacy.load("en_core_web_sm")
    # filename = "D:/NLPcoupling/dataset/trainingset_merged.pkl"
    # with open(filename, 'rb') as file_obj:
    #     training_d = pickle.load(file_obj)
    filtered_data = []
    for item in tqdm(rawdata):
        text = item[0]
        entities = item[1]['entities']
        doc = nlp.tokenizer(text)
        tags = spacy.gold.biluo_tags_from_offsets(doc, entities)
        if tags[-1] != '-':
            filtered_data.append(item)
    return filtered_data



if __name__ == '__main__':
    training_data = []
    datacons = dataConstruction()

    cterms = datacons.critical_terms
    wos_data = datacons.structured_data.sample(frac = 0.5).reset_index()  # frac = 0.2, frac = 0.5

    tic = time.time()
    pool = mp.Pool(6)
    subtrainingdata = pool.map(partial(trainingSet_generate, critical_terms=cterms), wos_data['TI_AB_CLEAN'].to_list())
    for item in subtrainingdata:
        training_data.extend(subtrainingdata)
    pool.close()
    toc = time.time()

    print("Time:{}".format(toc-tic))
    print("Length:{}".format(len(training_data)))
    pprint(training_data[:3])

    filtered_training_data = filter_data(training_data)
    filename = "D:/NLPcoupling/dataset/trainingset_0.8.json"
    with open(filename, 'w') as file_obj:
        json.dump(filtered_training_data, file_obj)
    print("Saved training data to json.")

    filename1 = "D:/NLPcoupling/dataset/trainingset_0.8.pkl"
    with open(filename1, 'wb') as file_obj:
        pickle.dump(filtered_training_data, file_obj)
    print("Saved training data to pickle.")

    filefortraining = r"D:\NLPcoupling\dataset\trainingset1_0.2.pkl"
    with open(filefortraining, 'rb') as fp:
        load_train_data = pickle.load(fp)
    tic = time.time()
    nerModeling(load_train_data)
    toc = time.time()

