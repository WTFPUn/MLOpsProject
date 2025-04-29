"""
the modek should be put on hugging face as opensource
target model is BAAI/bge-m3 from hugging face laded with FlagEmbedding library
https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3
"""
from FlagEmbedding import BGEM3FlagModel


def finetune(model_name = 'BAAI/bge-m3', train_file):
    # load current model
    model = BGEM3FlagModel(model_name,
                       use_fp16=True)
    # load new trainset
    # fit
    # we can call package FlagEmbedding.finetune.embedder.encoder_only.m3 for finetune model
    return model

def evaluate(model_name = 'BAAI/bge-m3', test_set):
    # load model
    model = BGEM3FlagModel(model_name,
                       use_fp16=True)
    # load data
    data = 
    # prediction
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


