import torch.nn as nn
import torch
from model.transformer import Encoder, Decoder, Transformer
import nltk.translate.bleu_score as bleu
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
import os
import pickle
import math

#===============================================================================================================================
# 번역(translation) 함수
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, logging=True):
    model.eval() # 평가 모드

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # 처음에 <sos> 토큰, 마지막에 <eos> 토큰 붙이기
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    if logging:
        print(f"전체 소스 토큰: {tokens}")

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    if logging:
        print(f"소스 문장 인덱스: {src_indexes}")

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # 소스 문장에 따른 마스크 생성
    src_mask = model.make_src_mask(src_tensor)

    # 인코더(endocer)에 소스 문장을 넣어 출력 값 구하기
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # 처음에는 <sos> 토큰 하나만 가지고 있도록 하기
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attention = 0
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # 출력 문장에 따른 마스크 생성
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # 출력 문장에서 가장 마지막 단어만 사용
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token) # 출력 문장에 더하기

        # <eos>를 만나는 순간 끝
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # 각 출력 단어 인덱스를 실제 단어로 변환
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # 첫 번째 <sos>는 제외하고 출력 문장 반환
    
    return trg_tokens[1:], attention

def evaluate_bleu(model, SRC, TRG ,dataset,device):
    model.eval() # 평가 모드
    bleu_score = 0
    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for data in dataset.examples:
            
            src = vars(data)['src']
            trg = vars(data)['trg']
            
            predict, _ = translate_sentence(src,SRC,TRG, model,device,logging = False)
            
            bleu_score += bleu.sentence_bleu([trg],predict[:-1])

            
    

    return bleu_score / len(dataset.examples)

# ======================================================================================================================================
def evaluate_ppl(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 출력 단어의 마지막 인덱스(<eos>)는 제외
            # 입력을 할 때는 <sos>부터 시작하도록 처리
            output, _ = model(src, trg[:,:-1])

            # output: [배치 크기, trg_len - 1, output_dim]
            # trg: [배치 크기, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0(<sos>)은 제외
            trg = trg[:,1:].contiguous().view(-1)

            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def testing_en_de_translation(args):
    
    print("--------------initialize model----------------")
    save_name = f'processed.pkl'
    
    with open(os.path.join(args.preprocess_path, save_name), 'rb') as f:
        data_ = pickle.load(f)
        SRC = data_['SRC']
        TRG = data_['TRG']
        del data_
    train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    
    
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 일반적인 데이터 로더(data loader)의 iterator와 유사하게 사용 가능
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        batch_size=args.batch_size,
        device=device)

    # 인코더(encoder)와 디코더(decoder) 객체 선언
    enc = Encoder(INPUT_DIM, args.d_embedding, args.num_encoder_layer, args.n_head, args.dim_enc_feedforward, args.enc_dropout, device)
    dec = Decoder(OUTPUT_DIM, args.d_embedding, args.num_decoder_layer, args.n_head, args.dim_dec_feedforward, args.dec_dropout, device)

    # Transformer 객체 선언
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    # Adam optimizer로 학습 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 뒷 부분의 패딩(padding)에 대해서는 값 무시
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    
    
    model.load_state_dict(torch.load(args.model_save_path + 'transformer_model.pt'))
    
    print("-------------- test start! ----------------")
    example_idx = 20

    src = vars(test_dataset.examples[example_idx])['src']
    trg = vars(test_dataset.examples[example_idx])['trg']

    print(f'소스 문장: {src}')
    print(f'타겟 문장: {trg}')

    translation, attention = translate_sentence(src, SRC, TRG, model, device, logging=True)

    print("모델 출력 결과:", " ".join(translation))
    test_loss = evaluate_ppl(model, test_iterator, criterion)
    print(f'\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')
    
    bleu_score = evaluate_bleu(model, SRC, TRG ,test_dataset,device)
    print(f'bleu SCORE : {bleu_score:.3f}')
    
def testing(args):
    if args.task == 'translation':
        testing_en_de_translation(args)