import torch
from torch import nn
import torch.nn.functional as F

class CBOW(nn.Module):
    """input  -- > batch_size x context_size
       output --> batch_size x vocabulary"""
    def __init__(self, vocabulary_Size, embedding_features, padding_idx=0):
        """nn.Embedding holds a tensor of dimmension (vocabulary_size, feature_size)-->N(0,1)"""
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = vocabulary_Size, embedding_dim = embedding_features, padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features = embedding_features, out_features = vocabulary_Size)
    def forward(self, x):
        x = self.embedding(x)
        x = x.sum(dim=1)/x.shape[1] 
        log_prob = F.log_softmax(self.fc1(x), dim=1).unsqueeze(1)
        return log_prob



class SKIP_GRAM(nn.Module):
    """x --> batch_size x word_index
       output --> batch_size x context_predicted x vocabulary"""    
    def __init__(self, vocabulary_Size, embedding_features, context_len, padding_idx=0 ):
        super(SKIP_GRAM, self).__init__()
        self.context_len = context_len
        self.embedding = nn.Embedding(num_embeddings = vocabulary_Size, embedding_dim=embedding_features, padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features = embedding_features, out_features = vocabulary_Size)
    
    def forward(self, x):
        context_out = []
        for i in range(self.context_len):
            x_ = self.embedding(x)
            context_word_i = self.fc1(x_)
            context_out.append(context_word_i)
        log_prob = F.log_softmax(torch.stack(context_out, dim=1).squeeze(), dim=1)
        return log_prob