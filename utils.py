import numpy as np
import nltk
import torch
from torch import nn
import torchtext
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import SpectralEmbedding


# check cuda availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Init TF-IDF and LSA models, LE
tfidf = TfidfVectorizer()
n_comp = 300 # Max components y_pred input data
lsa = TruncatedSVD(n_components=n_comp, algorithm="arpack")
le = SpectralEmbedding(n_components=300)
# Definition of Variables
vocab = None
text_pipeline = None

def yield_tokens(data_iter):
    for text in data_iter:
        yield nltk.word_tokenize(text)

def generate_vocabulary(data):
    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def generate_embedding_matrix(vocabulary: torchtext.vocab.Vocab, embed_model):
    embedding_matrix = np.zeros((vocabulary.__len__(), embed_model.vector_size))
    for word, index in vocabulary.get_stoi().items():
        try:
            embedding_matrix[index] = embed_model[word]
        except:
            continue
    return embedding_matrix

def binary_codes(target):
    """
    Convert source to binary code
    :param target: array of values to convert to binary codes
    :return: binary codes
    """
    median = np.median(target, axis=1).reshape((target.shape[0], 1))
    binary = np.zeros(shape=np.shape(target))
    binary[target > median] = 1
    return binary

def low_dimensional_vector(term_freq_mat, embedding_matrix=None, lwdv='LSA'):
    """
    Generate Low dimensional vector
    :param tokenizer: tokenizer object
    :param sequences_full: sequences of text
    :param embedding_matrix: embedded matrix
    :param lwdv: Type of low dimensional vector
    :return: low dimensional vector
    """
    if lwdv == 'LE':  # Laplacian Eigenmaps (LE) # MEMORY EXPENSIVE
            # explore parameters
        dim_reduction_le = le.transform(term_freq_mat)
        return dim_reduction_le
    elif lwdv == 'AE':  # Average embedding (AE) # MEMORY EXPENSIVE
        denom = 1 + np.sum(term_freq_mat, axis=1)[:, None]
        normed_tfidf = term_freq_mat / denom
        average_embeddings = np.dot(normed_tfidf, embedding_matrix)
        return average_embeddings
    elif lwdv == 'LSA':  # LSA
        dim_reduction_lsa = lsa.transform(term_freq_mat)
        return dim_reduction_lsa

def collate_batch_stc(batch: list):
    text_list = []
    B = []
    max_words_text = len(sorted(batch, key=len, reverse=True)[0].split())
    for _text in batch:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        processed_text = nn.functional.pad(processed_text, pad=(0, max_words_text-processed_text.shape[0])) #(padding_left,padding_right)
        text_list.append(processed_text)
    Y = low_dimensional_vector(tfidf.transform(batch)) # Low dimenstionality reduction LSA
    B = torch.tensor(binary_codes(Y), dtype=torch.float32)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return  B.to(device), text_list.to(device)
