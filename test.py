import re
import nltk
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from sklearn.model_selection import train_test_split

import utils
from models import CNN_STC
import train as tr


# Cleaning patterns
patterns = re.compile(r'([^\w\s]\s+)', flags=re.IGNORECASE)

def clean_abstract(text):
        text = patterns.sub(r' ', text)
        text = re.sub(r' +', ' ', text).strip()
        tk_text = nltk.word_tokenize(text)
        new_text = ['' if len(x) < 2 or (x.isnumeric() and len(x) > 4) else x.lower() for x in tk_text]
        return ' '.join(new_text)

# Read data
data_abstracts = pd.read_csv('data/arxiv_data_210930-054931.csv', sep=',')
# Clean data
data_abstracts['abs_cleaned'] = data_abstracts['abstracts'].apply(clean_abstract)
# Split data into training, validating, testing
x_train_idx, x_test_idx = train_test_split(data_abstracts['abs_cleaned'].index,test_size=0.3, shuffle=True, random_state=42)
x_val_idx, x_test_idx = train_test_split(x_test_idx, test_size=0.66, shuffle=True, random_state=42)

# Load Glove Model if needed
# glove_model = gensim.models.KeyedVectors.load_word2vec_format('data/glove.6B.50d.txt', binary=False, no_header=True)
# weights = torch.FloatTensor(glove_model.vectors)
# embed_matrix = generate_embedding_matrix(vocab, glove_model)

# Generate vocabulary
utils.vocab = utils.generate_vocabulary(data_abstracts['abs_cleaned'].values)
# Vocab + tokenization function
utils.text_pipeline = lambda x: utils.vocab(nltk.tokenize.word_tokenize(x))
# Fit TF-IDF and LSA
utils.tfidf.fit(data_abstracts["abs_cleaned"].values)
utils.lsa.fit(utils.tfidf.transform(data_abstracts["abs_cleaned"].values))
# Create DataLoaders
BATCH_SIZE = 25
train_dataloader_stc = DataLoader(data_abstracts.loc[x_train_idx]["abs_cleaned"].values, batch_size=BATCH_SIZE, shuffle=True, 
                                collate_fn=utils.collate_batch_stc)
test_dataloader_stc = DataLoader(data_abstracts.loc[x_test_idx]["abs_cleaned"].values, batch_size=BATCH_SIZE, shuffle=True, 
                                collate_fn=utils.collate_batch_stc)
validate_dataloader_stc = DataLoader(data_abstracts.loc[x_val_idx]["abs_cleaned"].values, batch_size=BATCH_SIZE, shuffle=True, 
                                        collate_fn=utils.collate_batch_stc)

# Define model
# vocab_size, embed_dim, filter_size, num_filter, num_classes, dropout
model_cnn_stc = CNN_STC(len(utils.vocab), utils.max_words_text, 4, 25, 300, 0.5)
print(summary(model_cnn_stc))
# Define Variables
tr.LR = 0.01 # By defult, customisable
tr.optimizer = torch.optim.Adam(model_cnn_stc.parameters(), lr=tr.LR)
tr.loss_function = torch.nn.BCELoss() # Input: (N, C), N=data, C=Total classes, Target:(N)
tr.train_model(epochs=1, train_dataloader=train_dataloader_stc, validation_data_loader=validate_dataloader_stc, model=model_cnn_stc)
