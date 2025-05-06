"""
the modek should be put on hugging face as opensource
target model is BAAI/bge-m3 from hugging face laded with FlagEmbedding library
https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3
"""
from FlagEmbedding import BGEM3FlagModel
import pandas as pd

def finetune(model_name = 'BAAI/bge-m3', train_file=""):
    # load current model
    model = BGEM3FlagModel(model_name,
                       use_fp16=True)
    # load new trainset
    # fit
    # we can call package FlagEmbedding.finetune.embedder.encoder_only.m3 for finetune model
    return model

def evaluate(model_name = 'BAAI/bge-m3', test_set=""):
    # load model
    model = BGEM3FlagModel(model_name,
                       use_fp16=True)
    # load data
    data = pd.read_csv(test_set)
    data.dropna(["context"],axis = 1)
    # prediction
    outputs = model.encode(sentences, return_dense=False, return_sparse=False, return_colbert_vecs=True)
    colbert_vecs = outputs['colbert_vecs']
    # compare to label
    return score

def push_to_hf(repo, model):
    """_summary_

    Args:
        repo (_type_): repo nae on huggingface
        model (_type_): model
    """
    model.push_to_hub(repo)
    pass


