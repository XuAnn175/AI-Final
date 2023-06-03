import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from bert import BERT, MailDataset
from ngram import Ngram
from preprocess import preprocessing_function

def prepare_data():
    df_train = pd.read_csv('train.tsv', sep='\t', header=None, names=['label', 'msg_body'])
    df_test  = pd.read_csv('test.tsv' , sep='\t', header=None, names=['label', 'msg_body'])
    return df_train, df_test

def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--model_type",
                        type=str,
                        choices=['ngram', 'BERT'],
                        required=True,
                        help="model type")
   
    opt.add_argument("--part",
                        type=int,
                        help="specify the part")

    opt.add_argument("--N",
                        type=int,
                        help="uni/bi/tri-gram",
                        default=2)
    opt.add_argument("--num_features",
                        type=int,
                        help="number of features",
                        default=500)
    config = vars(opt.parse_args())
    return config

def GetNgramModel(model_type, df_train, df_test, config, N):
    # load and train model
    model = Ngram(config = config, n = N)
    model.train(df_train)
    return model

def second_part(model_type, df_train, df_test, config, N):# model_type = ngram
    # load model
    model = GetNgramModel(model_type, df_train, df_test, config, N)
    # train model
    model.train_label(df_train, df_test)

def fourth_part(model_type, df_train, df_test, config, N):
    # model_type = bert
    bert_config = {# configurations
        'batch_size': 8,
        'epochs': 1,
        'lr': 2e-5,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    }
    train_data = MailDataset(df_train)
    test_data = MailDataset(df_test)
    train_dataloader = DataLoader(train_data, batch_size=bert_config['batch_size'])
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = BERT('distilbert-base-uncased', bert_config)
    model.train_label(train_dataloader, test_dataloader)

if __name__ == '__main__':
    # get argument
    config = get_argument()
    model_type = config['model_type']
    N = config['N']                      

    # read and prepare data
    df_train, df_test = prepare_data()

    label_mapping = {'spam': 0, 'ham': 1}
    df_train['label'] = df_train['label'].map(label_mapping)
    df_test['label']  = df_test['label'].map(label_mapping)
    
    # preprocessing
    df_train['msg_body'] = df_train['msg_body'].apply(preprocessing_function)
    df_test['msg_body']  = df_test['msg_body'].apply(preprocessing_function)

    if config['part'] == 1: 
        GetNgramModel(model_type, df_train, df_test, config, N)
    elif config['part'] == 2:
        # Part 2: build model using sorting feature selection and calculate the corresponding f1 score,precision,recall
        second_part(model_type, df_train, df_test, config, N)
    elif config['part'] == 3:
        # Part 3: build model using chi square feature selection and calculate the corresponding f1 score,precision,recall
        second_part(model_type, df_train, df_test, config, N)
    elif config['part'] == 4:
        # Part 4 : bert
        fourth_part(model_type, df_train, df_test, config, N)
    

