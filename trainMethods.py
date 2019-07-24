import torch
import matplotlib.pyplot as plt
import math

#press ctrl+c to stop traning and see the plot till the last epoch.
def plot_error(train_error, val_error,last_epoch):
    """train loss and validation loss plot"""
    x_val = torch.arange(0,last_epoch,10)
    plt.plot(torch.tensor(train_error).numpy())
    plt.plot(x_val.numpy(),torch.tensor(val_error).numpy())
    plt.show()

def save_model(model, least_val, current_val, filename, epoch):
    """saves the best model till current epoch"""
    if least_val > current_val:
        torch.save(model.state_dict(), filename[:-4]+f"_epoch-{epoch}.pth")
        return current_val
    return least_val

def train(**kwargs) :
    """ model --> model
        optimizer --> optimizer used
        loss_function -->  loss function used
        train_iter --> training dataset itterator
        val_itter --> validation dataset itterator
        epoch --> number of epochs
        model_type --> CBOW for contineous bag of words/ SKIP GRAM for skip gram
        filename --> inital name of file that has the best model.parametes() till current epoch
        val_interval --> interval at will validation needs to be done
        negetive_sample --> True is Negetive sample required
        "ns" --> number of negetive samples to take
        """
    #paramters -->
    parameters = ["model","optimizer","loss_function", "train_iter", "val_iter","epoch", \
                            "model_type", "filename", "val_interval","Vocabulary","negetive_sample", "ns"]
    model,optimizer,loss_function, train_iter, val_iter,epoch, \
                            model_type, filename, val_interval, Vocab, negetive_sample, ns=\
                                                                            [kwargs[i] for i in parameters]
    optimizer = optimizer
    loss_function = loss_function
    train_error, val_error = [], []
    least_val = math.inf
    try:
        for epoch_i in range(epoch):
            model.train()
            rolling_loss = 0
            for context_i, word_i in train_iter:
                
                if model_type == "SKIP GRAM":
                    log_prob = model(word_i)
                    loss=1
                    for i in range(log_prob.shape[1]):
                        loss_i = loss_function(log_prob[:,i,], context_i[:,i])
                        loss = loss*loss_i
                    loss = loss/(i+1)
                else:
                    log_prob = model(context_i)
                    if negetive_sample == True: 
                        context_i = neg_sample(log_prob.squeeze(), ns , Vocab)
                    loss = loss_function(log_prob.squeeze(), word_i.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rolling_loss += loss
            train_error.append(rolling_loss)
            print ("training loss in epoch: ",epoch_i,"-->", rolling_loss.item())
            rolling_val = 0
            #validation below
            if epoch_i %val_interval == 0:
                model.eval()
                for context_i, word_i in val_iter:
                    if model_type == "SKIP GRAM":
                        log_prob = model(word_i)
                        loss_val=1
                        for i in range(log_prob.shape[1]):
                            loss_i = loss_function(log_prob[:,i,], context_i[:,i])
                            loss_val = loss_val*loss_i
                        loss = loss/(i+1)
                    else:
                        log_prob = model(context_i)
                        loss_val = loss_function(log_prob.squeeze(), word_i.squeeze())
                    rolling_val += loss_val
                least_val = save_model(model,least_val, rolling_val.mean().item(), filename, epoch_i)
                val_error.append(rolling_val)
                print ("validation loss in epoch: ",epoch,"-->", rolling_val.mean().item())
        plot_error(train_error, val_error, epoch_i )
    except KeyboardInterrupt:
        plot_error(train_error, val_error, epoch_i )

def neg_sample(context, negSample_size, Vocabulary):
    prob_selection = Vocabulary.neg_samples()
    index, prob = map(torch.tensor,[list(prob_selection.keys()), list(prob_selection.values())])
    topk_index = torch.topk(prob,k=5)[1]
    index_To_take = index[topk_index]
    new_context = torch.stack([context[i, index_To_take] for i in range(context.shape[0])])
    return new_context.unsqueeze(1)