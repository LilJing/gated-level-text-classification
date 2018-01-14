# coding=utf-8
import torch
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import word2vec
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
import numpy as np
import pandas as pd
import csv
Ag_model = gensim.models.Word2Vec.load('/home/jli/project/char/Agmodel.model')
Ag_dict=Dictionary.load('/home/jli/project/char/Ag_dict.dict')
# initialize unk
from compiler.ast import flatten
unk=np.random.randn(1,200)
unk=unk.flatten()
unk=np.array(unk)
vocab=[]
for id, word in enumerate(Ag_dict.token2id):
    vocab.append(word.encode())
# data_path=r'/home/jli/project/ag_news_csv/train.csv'
class AGNEWs():
    def __init__(self, data_path):
        self.data_path = data_path
        self.load()
        self.y = torch.LongTensor(self.label)

    def __getitem__(self, idx):
        data_txt = self.word_emb(idx)
        y = self.y[idx]
        return data_txt, y

    def __len__(self):
        return len(self.label)
    def load(self):
        self.label = []
        self.data = []
        with open(self.data_path, 'rb') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for index, row in enumerate(rdr):
                self.label.append(int(row[0]))
                txt = ' '.join(row[1:])
                txt = txt.lower()
                txt = str(txt.split())
                txt = txt.translate(None, '[\+\.\9876543210!\/_,==-;$%^*()\+\"\']:+|[+——！，。?？、~@#￥%……&*（）]+')
                txt = txt.split()
                self.data.append(txt)

    def word_emb(self, idx):
        sequence = self.data[idx]
        # print len(sequence)
        data_txt=torch.zeros(30,200)
        # print data_txt
        data_txt1=[]
        for w in sequence:
            if w in vocab:
                data_txt1.append(Ag_model[w])
            else:
                data_txt1.append(unk)
        for i in range(min(len(sequence),30)):
            for j in range(200):
                data_txt[i][j] = np.float64(data_txt1[i][j]).item()
        return data_txt.transpose(0,1)

if __name__ == '__main__':
    main()