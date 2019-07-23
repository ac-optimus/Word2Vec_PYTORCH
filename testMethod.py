import torch
from torch import nn, optim

def test(**kwargs):
    """ model --> model
        loss_function -->  loss function used
        test_itter --> test dataset itterator
        model_type --> CBOW for contineous bag of words/ SKIP GRAM for skip gram
        Vocabulary --> Vocabulary of token and index
        """
    #parameter -->
    parameters = ["model","loss_function", "test_iter","model_type","Vocabulary"]
    model,loss_function, test_iter, model_type, Vocabulary=\
                                          [kwargs[i] for i in parameters]
    model.eval()
    y_hat = []
    rolling_error = 0
    for context_i, word_i in test_iter:
        # if model_type = "SKIP GRAM":
            
        if model_type == "SKIP GRAM":
            log_prob = model(word_i)
            loss=1
            for i in range(log_prob.shape[1]):
                loss_i = loss_function(log_prob[:,i,], context_i[:,i])
                loss = loss*loss_i
            loss = loss/(i+1)
        else:
            log_prob = model(context_i)
            loss = loss_function(log_prob.squeeze(), word_i.squeeze())
        
        out_index = log_prob.max(dim=2)[1]
        
        y_hat_i = []
        for batch_i in range(out_index.shape[0]):
            batch_i_yhat = []
            for word_i in range(out_index.shape[1]):
                batch_i_yhat.append( Vocabulary.get_token_for_index(out_index[batch_i,word_i].item()))
            y_hat_i.append(batch_i_yhat)
        rolling_error += loss
    y_hat.append(y_hat_i)
    # print ("the value of x-->",context_i," the value of y_hat-->",y_hat)
    return y_hat
      