# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
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
from compiler.ast import flatten
unk=np.random.randn(1, 200)
unk=unk.flatten()
unk=np.array(unk)
vocab=[]
for id, word in enumerate(Ag_dict.token2id):
    vocab.append(word.encode())
# data_path=r'/home/jli/project/ag_news_csv/train.csv'
class AGNEWs():
    def __init__(self, label_data_path, alphabet_path):

        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))

        self.alphabet = alphabet
        self.label_data_path = label_data_path
        self.l0 = 200
        self.load()
        self.y = torch.LongTensor(self.label)

    def __getitem__(self, idx):
        data_txt1 = self.get_char_list(idx)
        data_txt2 = self.get_word_emb(idx)
        y = self.y[idx]
        return data_txt1,data_txt2, y

    def __len__(self):
        return len(self.label)

    def load(self):
        self.label = []
        self.data = []
        with open(self.label_data_path, 'rb') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for index, row in enumerate(rdr):
                self.label.append(int(row[0]))
                txt = ' '.join(row[1:])
                txt = txt.lower()
                txt = str(txt.split())
                txt = txt.translate(None, '[\+\.\9876543210!\/_,==-;$%^*()\+\"\']:+|[+——！，。?？、~@#￥%……&*（）]+')
                txt = txt.split()
                data_txt = []
                self.data.append(txt)

    def get_char_list(self,idx):
        X = torch.zeros(32, 6).long()
        sequence=self.data[idx]
        for w_id, w in enumerate(sequence):
            if w_id>31:
                break
            if len(w)<=6:
                for char_id, char in enumerate(w):
                    # X[w_id][char_id+6*w_id]=self.char2Index(char)
                    X[w_id][char_id] = self.char2Index(char)
            else:
                w_minus=w[0:6]
                for char_id, char in enumerate(w_minus):
                    # X[w_id][char_id+6*w_id]=self.char2Index(char)
                    X[w_id][char_id] = self.char2Index(char)
        return X

    def get_word_emb(self,idx):
        sequence = self.data[idx]
        data_txt_word = torch.zeros(32, 200)
        data_txt1 = []
        for w in sequence:
            if w in vocab:
                data_txt1.append(Ag_model[w])
            else:
                data_txt1.append(unk)
        for i in range(min(len(sequence), 32)):
            for j in range(200):
                data_txt_word[i][j] = np.float64(data_txt1[i][j]).item()
        data_txt_word = data_txt_word.transpose(0, 1)
        return data_txt_word


    def char2Index(self, character):
        return self.alphabet.find(character)

if __name__ == '__main__':
    main()