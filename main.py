from models import CBOW, SKIP_GRAM
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from trainMethods import train
from testMethod import test
from my_classes import *

CONTEXT_LEN = 5
FILENAME = "dummy_corpus.txt"  #replace with your corpus file
BATCH_SIZE = 2

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
# model = CBOW(100,20)
model = SKIP_GRAM(100,20,context_len=10)

#train --> uncomment accordingly
optimizer = optim.Adam(model.parameters(), lr = 0.001 )
loss_function = nn.NLLLoss()
# train(model, optimizer, loss_function, train_loader,val_loader,10,"CBOW", filename="", val_interval=10)
train(model, optimizer, loss_function, train_loader, val_loader,10,"SKIP GRAM", filename="", val_interval=10)

#test --> uncomment accordingly
# y_hat = test(model, loss_function,test_loader, Vocab, model_type = "CBOW")
y_hat = test(model, loss_function,test_loader, Vocab, model_type = "SKIP GRAM")
# print (y_hat)
