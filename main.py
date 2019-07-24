from models import CBOW, SKIP_GRAM
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from trainMethods import train
from testMethod import test
from my_classes import *

CONTEXT_LEN = 5
FILENAME = "dummy_corpus.txt"  #replace with your corpus file
BATCH_SIZE = 200

#get the datsaet
with open(FILENAME, 'r') as f:
    corpus = f.read()
# print (corpus)

#add tokenization step here
preprocess = preprocessing()
token = preprocess.tokenize(corpus)

#setup the vocabulary
Vocab =Vocabulary(token)
token_to_index = Vocab.token_to_index

data = preprocess.get_context(CONTEXT_LEN, corpus)
preprocess.convet_to_word_index (token_to_index, data)

trainset, valset, testset = preprocess.split_train_val_test((0.7,0.15,0.15), data)
print ("lenght of: dataset-->",len(data), \
        "train-->",len(trainset), "val-->",len(valset), "test-->",len(testset))

#get the dataset ready for train, val and test
train_set = DatasetNLP(trainset)
val_set = DatasetNLP(valset)
test_set = DatasetNLP(testset)
x,y  = train_set.get_dataset()
# # x_i, y_i = dataset.__getitem__(0)
print ("dimmension of x: ", x.shape)
print ("dimmeinson of y: ", y.shape)

train_loader = DataLoader(dataset=train_set, \
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_set, \
                        batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set,\
                         batch_size=BATCH_SIZE, shuffle=True)

# choose the model-->uncomment one that you want to play with
num_embeddings, embedding_dim = len(Vocab), 300
model = CBOW(num_embeddings, embedding_dim )
# model = SKIP_GRAM(100,20,context_len=10)
print ("reached here")
#train -->
optimizer = optim.Adam(model.parameters(), lr = 0.001 )
loss_function = nn.NLLLoss()
setting = {"model":model,
            "optimizer" : optimizer,
            "loss_function":loss_function,
            "train_iter":train_loader,
            "val_iter":val_loader,
            "test_iter":test_loader,
            "epoch":10,
            "model_type":"CBOW",
            # "model_type":"SKIP GRAM",
            "filename":"",
            "val_interval":10,
            "Vocabulary":Vocab,
            "ns":5,
            "negetive_sample":False}
train(**setting)

#test -->
y_hat = test(**setting)
print (y_hat)