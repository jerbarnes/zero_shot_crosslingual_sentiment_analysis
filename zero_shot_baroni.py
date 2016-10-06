import sys
sys.path.append('/home/jeremy/NS/Keep/Temp/Exps/BilBowa_Exps/')
sys.path.append('/home/jeremy/NS/Keep/Temp/Exps/Mikolov_Exps/')
sys.path.append('/home/jeremy/Escritorio/zero_shot_experiment/transmat')
import numpy as np
from SpanishCrosslingualExperiment import test_models
from get_n_best_results import *
from transmat.utils import*
from transmat.space import Space
from Datasets import General_Dataset, Spanish_Dataset, English_Dataset
from Representations import getMyData
from MyMetrics import *
from keras_parameters_search import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.regularizers import l1, l2


def average_vec(sent, model):
    vec_size = model.mat.shape[1]
    sent_len = len(sent)
    sent_vec = np.zeros((1, vec_size))
    for w in sent.split():
        try:
            sent_vec += model.mat[model.row2id[w]]
        except KeyError:
            pass
            sent_len -= 1
    sent_vec /= len(sent)
    return sent_vec

def get_lexicon(DIR):
    print('getting lexicon')
    pos =[[w.lower() for w in sent.split()]
      for sent in open(DIR+'/pos.txt').readlines()]
    strong_pos = [[w.lower() for w in sent.split()]
      for sent in open(DIR+'/strpos.txt').readlines()]
    strong_neg = [[w.lower() for w in sent.split()]
      for sent in open(DIR+'/strneg.txt').readlines()]
    neg = [[w.lower() for w in sent.split()]
      for sent in open(DIR+'/neg.txt').readlines()]
    lexicon = set([w for sent in strong_pos+pos+neg+strong_neg for w in sent])

    return lexicon

def train_mapping_model(spanish_embedding_file, english_embedding_file):
    print('Training translation matrix')
    train_data = read_dict('./data/train_sp_en')

    source_words, target_words = zip(*train_data)
    
    print("Reading: %s" % spanish_embedding_file)
    source_sp = Space.build(spanish_embedding_file, set(source_words))
    source_sp.normalize()

    print("Reading: %s" % english_embedding_file)
    target_sp = Space.build(english_embedding_file, set(target_words))
    target_sp.normalize()

    print("Learning the translation matrix")
    tm = train_tm(source_sp, target_sp, train_data)

    print("Printing the translation matrix")
    np.savetxt("%s.txt" % 'tm', tm)
    

def get_mapping_model(spanish_embedding_file, english_embedding_file):
    print('Building Spanish model')
    spanish_lexicon = get_lexicon('/home/jeremy/NS/Keep/Permanent/Corpora/OpeNER_Data/spanish')
    spanish_sp = Space.build(spanish_embedding_file,
                             spanish_lexicon, encoding='latin1')
    spanish_sp.normalize()

    print('Building English model')
    english_sp = Space.build(english_embedding_file, encoding='latin1')
    english_sp.normalize()

    print('Building Mapping model')
    tm = np.loadtxt('tm.txt')
    mapping_sp = apply_tm(spanish_sp, tm)
    return english_sp, spanish_sp, mapping_sp

def test_mapping_without_search(english_sp, spanish_sp, mapping_sp):

    spanish_dataset = Spanish_Dataset(spanish_sp, rep=average_vec)
    english_dataset = English_Dataset(english_sp, rep=average_vec)
    crosslingual_dataset = Spanish_Dataset(mapping_sp, rep=average_vec)

    english_vec_size = english_sp.mat.shape[1]
    spanish_vec_size = spanish_sp.mat.shape[1]
    
    english_mono_parameters = [[(english_vec_size, 1000, 'relu'), (1000, 1200, 'relu'),  (1200, 4, 'softmax')],
                          Sequential,
                          'categorical_crossentropy',
                          'Adam', 5, False]
    Spanish_mono_parameters = [[(spanish_vec_size, 1000, 'relu'), (1000, 1200, 'relu'), (1200, 4, 'softmax')],
                          Sequential,
                          'categorical_crossentropy',
                          'Adam', 5, False]

    test_models(english_mono_parameters,
                Spanish_mono_parameters,
                english_dataset,
                spanish_dataset,
                crosslingual_dataset)

    return spanish_sp, english_sp
    
if __name__ == '__main__':

    spanish_embedding_file = sys.argv[1]
    english_embedding_file = sys.argv[2]
    
    train_mapping_model(spanish_embedding_file, english_embedding_file)
    english_sp, spanish_sp, mapping_sp = get_mapping_model(spanish_embedding_file, english_embedding_file)
    spanish_dataset = Spanish_Dataset(spanish_sp, rep=average_vec)
    english_dataset = English_Dataset(english_sp, rep=average_vec)
    crosslingual_dataset = Spanish_Dataset(mapping_sp, rep=average_vec)
    test_mapping_without_search(english_sp, spanish_sp, mapping_sp)
