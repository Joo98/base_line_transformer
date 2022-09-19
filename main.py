import argparse
from preprocessing import preprocessing
from training import training
from testing import testing
import time

HIDDEN_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
SRC_PAD_IDX = None
TRG_PAD_IDX = None
N_EPOCHS = 10
CLIP = 1

ATTN_OPTION = 'CT'

def main(args):

    # Time setting
    total_start_time = time.time()

    print("start")

    if args.preprocessing:
        preprocessing(args)

    if args.training:
        training(args)

    if args.testing:
        testing(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path setting
    parser.add_argument('--preprocess_path', default='C:/Users/user/OneDrive/문서/NLP/base_line_transformer/preprocessed', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--data_path', default='/mnt/md0/dataset', type=str,
                        help='Original data path')
    parser.add_argument('--model_save_path', default='model_save/', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--result_path', default='./result', type=str,
                        help='Results file path')
    
    # Preprocessing setting
    parser.add_argument('--tokenizer', default='spm_bpe', choices=[
        'spm_unigram', 'spm_bpe', 'spm_word', 'spm_char', 'alpha_map'
            ], help='Tokenizer select; Default is spm')
    parser.add_argument('--character_coverage', default=1.0, type=float,
                        help='Source language chracter coverage ratio; Default is 1.0')
    parser.add_argument('--pad_id', default=0, type=int,
                        help='Padding token index; Default is 0')
    parser.add_argument('--unk_id', default=3, type=int,
                        help='Unknown token index; Default is 3')
    parser.add_argument('--bos_id', default=1, type=int,
                        help='Padding token index; Default is 1')
    parser.add_argument('--eos_id', default=2, type=int,
                        help='Padding token index; Default is 2')
    
    # Model setting
    # 0) Model selection
    parser.add_argument('--model_type', default='custom_transformer', type=str, choices=[
        'custom_transformer', 'bart', 'T5', 'bert'
            ], help='Model type selection; Default is custom_transformer')
    #parser.add_argument('--isPreTrain', default=False, type=str2bool,
    #                    help='Using pre-trained model; Default is False')
    
    # 1) Custom Transformer
    parser.add_argument('--d_model', default=768, type=int, 
                        help='Transformer model dimension; Default is 768')
    parser.add_argument('--d_embedding', default=HIDDEN_DIM, type=int, 
                        help=f'Transformer embedding word token dimension; Default is {HIDDEN_DIM}')
    parser.add_argument('--n_head', default=ENC_HEADS, type=int, 
                        help="Multihead Attention's head count; Default is 12")
    parser.add_argument('--dim_enc_feedforward', default=ENC_PF_DIM, type=int, 
                        help=f"Feedforward network's dimension; Default is {ENC_PF_DIM}")
    parser.add_argument('--dim_dec_feedforward', default=DEC_PF_DIM, type=int, 
                        help=f"Feedforward network's dimension; Default is {DEC_PF_DIM}")
    parser.add_argument('--enc_dropout', default=ENC_DROPOUT, type=float, 
                        help=f"Dropout ration; Default is {ENC_DROPOUT}")
    parser.add_argument('--dec_dropout', default=DEC_DROPOUT, type=float, 
                        help=f"Dropout ration; Default is {DEC_DROPOUT}")
    parser.add_argument('--embedding_dropout', default=0.1, type=float, 
                        help="Embedding dropout ration; Default is 0.1")
    parser.add_argument('--num_encoder_layer', default=ENC_LAYERS, type=int, 
                        help=f"Number of encoder layers; Default is {ENC_LAYERS}")
    parser.add_argument('--num_decoder_layer', default=DEC_LAYERS, type=int, 
                        help=f"Number of decoder layers; Default is {DEC_LAYERS}")
    
    parser.add_argument('--attn_option', default=ATTN_OPTION, type=str,
                        help='attention task')
    
    parser.add_argument('--num_common_layer', default=8, type=int, 
                        help="Number of common layers; Default is 8")
    
    # Optimizer & LR_Scheduler setting
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, 
                        help=f"Embedding dropout ration; Default is {LEARNING_RATE}")

    # Training setting
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int,    
                        help=f'Batch size; Default is {BATCH_SIZE}')
    parser.add_argument('--clip', default=CLIP, type=int, 
                        help=f"Number of common layers; Default is {CLIP}")
    parser.add_argument('--num_epochs', default=N_EPOCHS, type=int, 
                        help=f"Number of common layers; Default is {N_EPOCHS}")
    #============================additional args
    parser.add_argument('--task', default='translation', type=str,
                        help='Model task')
    parser.add_argument('--data_name', default='en_de_Multi30k', type=str,
                        help='Model data_name')
    
    
    args = parser.parse_args()
    
    main(args)