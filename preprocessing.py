import os
import time
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import pickle
import logging


spacy_en = spacy.load('en_core_web_sm') # 영어 토큰화(tokenization)
spacy_de = spacy.load('de_core_news_sm') # 독일어 토큰화(tokenization)

# 독일어(Deutsch) 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

def preprocess_for_en_de_Multi30k(args):
    print("------------ start preprocessing --------------")
    SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
    
    tokenized = spacy_en.tokenizer("I am a graduate student.")
    print('example of tokenized sentence for // I am a graduate student.')
    for i, token in enumerate(tokenized):
        print(f"인덱스 {i}: {token.text}")
    
    save_name = f'processed.pkl'

    with open(os.path.join(args.preprocess_path, save_name), 'wb') as f:
        pickle.dump({
            'SRC' : SRC,
            'TRG' : TRG
        }, f)
        
    
    
def preprocessing(args):
    
    if args.task == 'translation':
        preprocess_for_en_de_Multi30k(args)
    