import torch
from torch.utils.data import Dataset
class Vocabulary:
    """takes preprocessed tokens and build the Vocabulary"""
    def __init__(self, lst_tokens = []):
        self.token_to_index = {}
        self.token_to_index = {word_i:i for i, word_i in enumerate(set(lst_tokens))}
        self.index_to_token  = {}
        self.index_to_token = {i:word_i for i, word_i in enumerate(set(lst_tokens))}
        # self.types = list(set(lst_tokens))

        # for i in range(len(self.types)):
        #     if self.types[i] not in self.token_to_index.keys():
        #         self.token_to_index[self.types[i]] = i
        #         self.index_to_token[i] = self.types[i]
    
    def add_token(self, token):
        """adds a new element to the existing Vocabulary"""
        if token in self.token_to_index.keys():
            pass 
        else:
            index_i = len(self.token_to_index)
            self.token_to_index[token]  = index_i
            self.index_to_token[index_i] = token
    
    def get_index_for_token(self, token):
        """returns index for the passed token(str)"""
        if token in self.token_to_index.keys():
            return self.token_to_index[token]
        return ("token not in Vocabulary")
    
    def get_token_for_index(self, index):
        """return token for a given index"""
        if index in self.index_to_token.keys():
            return self.index_to_token[index]
        return "No token with this index"
    
    def __len__(self):
        """returns the Vocabulary size"""
        return len(self.token_to_index)

class preprocessing:
    def __init__(self):
        pass
    def tokenize(self, corpus):
        """tokenize the given corpus which can be an itterator(in future)"""
        return corpus.split()

    def get_context(self,CONTEXT_LEN, corpus):
        token = corpus.split()
        context = []
        for i in range(1,len(token)-1):
            if i<CONTEXT_LEN :
                lst_context = [token[0]]*(CONTEXT_LEN-i)+token[:i]+ token[i+1 : i+CONTEXT_LEN+1]
            elif i+CONTEXT_LEN >len(token):
                lst_context = token[i-CONTEXT_LEN : i] + token[i+1 : len(token)] +[token[-1]]*((i+CONTEXT_LEN) -len(token)+1)
            context.append([lst_context,token[i]])
        return context

    def convet_to_word_index(self,word_location, context_lst):
        for i in range(len(context_lst)):
            context_lst[i][0] = list(map(lambda x: word_location[x], context_lst[i][0]))
            context_lst[i][1] = word_location[context_lst[i][1]]
        return context_lst
    
class DatasetNLP(Dataset):
    def __init__(self,context_lst, transdorms=None):
        self.data = context_lst
        self.x, self.y = self.get_dataset()
    
    def __len__(self):
        return len(self.data)
            
    def get_dataset(self):
        x, y=[],[]
        lst = []
        for i, j in self.data:
            lst.append(torch.tensor(i, dtype = torch.long))
            y.append(j)
        x = torch.stack(lst)
        y = torch.tensor(y).unsqueeze(1)
        return x,y 
        
    def __getitem__(self,index):
        return self.x[index], self.y[index]