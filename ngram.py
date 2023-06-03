from itertools import count
from logging.config import stopListening    
from collections import defaultdict, OrderedDict
from pydoc import doc
from typing import List
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
import numpy as np

class Ngram:
    def __init__(self, config=None, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        unigramCnt = defaultdict(int)
        bigramCnt  = defaultdict(int)
        trigramCnt = defaultdict(int)
        bigrams    = []
        unigrams   = []
        trigrams   = []
        total = 0
        for document in corpus_tokenize:
            for idx in range(len(document)-1):
                total += 1
                if idx != len(document) - 2:
                    trigram = (document[idx], document[idx+1], document[idx+2]) # Generate trigram
                    trigramCnt[trigram] += 1
                    trigrams.append(trigram)
                bigram = (document[idx], document[idx+1])
                unigrams.append(document[idx])
                bigrams.append(bigram)

                bigramCnt[bigram] += 1
                unigramCnt[document[idx]] += 1
        
        if self.n != 1:
            model = defaultdict(lambda: defaultdict(int)) 
        else:
            model = defaultdict(int) 

        self.tt = total
        if self.n == 1: # unigram
            for uni in unigrams:
                model[uni] += unigramCnt[uni]
            
            features = OrderedDict(sorted(unigramCnt.items(), key=lambda item: -item[1])) # Higher appearance,lower index
        elif self.n == 2: # bigram
            for bigram in bigrams:
                x = bigram[0]
                y = bigram[1]            
                model[x][y] += 1
            
            features = OrderedDict(sorted(bigramCnt.items(), key=lambda item: -item[1]))
        elif self.n == 3: # trigram
            for trigram in trigrams:
                x = trigram[0]
                y = trigram[1]
                z = trigram[2]

                model[(x, y)][z] += 1
            features = OrderedDict(sorted(trigramCnt.items(), key=lambda item: -item[1]))

        self.unigramCnt = unigramCnt
        self.bigramCnt  = bigramCnt
        self.trigramCnt = trigramCnt

        return model, features
    
    def train(self, df): # model,data_to_be_trained
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['msg_body']]     # [CLS] represents start of sequence
        self.model, self.features = self.get_ngram(corpus)
  
    def get_chi2_features(self, df_train):
        feature_num = self.config['num_features'] # Get number of features from self.config
        gramPos = {}
        gramNeg = {}
        sumPos = 0
        sumNeg = 0
        train_corpus = [['[CLS]'] + self.tokenize(document) for document in df_train['msg_body']]
        y = list(df_train['label'])
        numGram = self.n
    
        for i, document in (enumerate(train_corpus)):
            for idx in range(len(document) - numGram + 1):
                #cur = Non
                if(numGram == 1):
                    cur = document[idx]
                elif(numGram == 2):   
                    cur = (document[idx], document[idx+1])
                elif(numGram == 3): 
                    cur = (document[idx], document[idx+1], document[idx+2])
                if y[i] == 1:
                    if cur in gramPos:
                        gramPos[cur] += 1
                    else:
                        gramPos[cur] = 1
                    sumPos += 1
                else:
                    if cur in gramNeg:
                        gramNeg[cur] += 1    
                    else:
                        gramNeg[cur] = 1
                    sumNeg += 1
        
        sumAll = sumPos + sumNeg
        all = {**gramPos, **gramNeg}
        chiFeatures = []
        for key in all:
            cur = 0
            if key in gramPos:
                cur += gramPos[key]
            if key in gramNeg:
                cur += gramNeg[key]
            Expected11 = sumPos * cur / sumAll
            Expected10 = sumNeg * cur / sumAll
            Expected01 = sumPos * (sumAll-cur) / sumAll
            Expected00 = sumNeg * (sumAll-cur) / sumAll
            chi =  ((gramPos[key] if key in gramPos else 0) - Expected11)**2 / Expected11
            chi += ((gramNeg[key] if key in gramNeg else 0) - Expected10)**2 / Expected10
            chi += (sumPos - (gramPos[key] if key in gramPos else 0) - Expected01)**2 / Expected01
            chi += (sumNeg - (gramNeg[key] if key in gramNeg else 0) - Expected00)**2 / Expected00
            chiFeatures.append((key, chi))
        chiFeatures = sorted(chiFeatures, key=lambda pair: -pair[1])
        kBestFeatures = chiFeatures[:feature_num]
        return kBestFeatures

    def train_label(self, df_train, df_test):
        feature_num = self.config['num_features']
        self.train(df_train) 
        gramIdx = {}
        train_corpus = [['[CLS]'] + self.tokenize(document) for document in df_train['msg_body']]

        if(self.config['part'] == 2):
            kBestFeatures = list(self.features.items())[:feature_num] # Use uni-gram that has highest freq. as features
        else:
            kBestFeatures = self.get_chi2_features(df_train)          # Use chi squre feature selection
        print(kBestFeatures[:20])
        
        test_corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['msg_body']] 

        train_corpus_embedding = [[0] * len(kBestFeatures) for _ in range(len(df_train['msg_body']))]
        test_corpus_embedding  = [[0] * len(kBestFeatures) for _ in range(len(df_test['msg_body']))]
        
        for i, pair in enumerate(kBestFeatures):
            gramIdx[pair[0]] = i 
        numGram = self.n
        # Convert corpus to embedding
        for i, document in enumerate(train_corpus):
            for idx in range(len(document) - numGram + 1):
                cur = None
                if(numGram == 1):
                    cur = document[idx]
                elif(numGram == 2):   
                    cur = (document[idx], document[idx+1])
                elif(numGram == 3): 
                    cur = (document[idx], document[idx+1], document[idx+2])
                if cur in gramIdx:
                    train_corpus_embedding[i][gramIdx[cur]] += 1
        
        for i, document in enumerate(test_corpus):
            for idx in range(len(document) - numGram + 1):
                cur = None
                if(numGram == 1):
                    cur = document[idx]
                elif(numGram == 2):   
                    cur = (document[idx], document[idx+1])
                elif(numGram == 3): 
                    cur = (document[idx], document[idx+1], document[idx+2])
                if cur in gramIdx:
                    test_corpus_embedding[i][gramIdx[cur]] += 1
        
        # feed converted embeddings to Naive Bayes
        clf = GaussianNB()
        clf.fit(train_corpus_embedding, df_train['label'])
        y_predicted = clf.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['label'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")

 
if __name__ == '__main__':
    test_sentence = {'msg_body': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
