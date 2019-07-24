
# Word2Vec_PYTORCH

The repository contains PyTorch implementation of Word2Vec. All the major tasks like vocabulary creation, preprocessing, 
train and test methods are supported. This can be helpful in any NLP project where Word2Vec is an essential part. 
The PyTorch implementation enables simple tweaks that can be useful. 

## **Repository Structure:** 

### models.py
- [x] ```CBOW```-Contineous bag of words
- [x] ```SKIP-GRAM```

### trainMethods.py
 ##### Generalized structure for training word2vec/ any neural network
- [x] ```plot_error``` - Can stop training at any epoch on keyboard interruption and plot the trian v/s validation error.
- [x] ```save_model```- Save the best model till current epoch.
- [x] ```train``` - Training method.
- [ ] negative sampling
- [ ] subsampling

### my_classes.py
-  Vocabulary <br />
   - [x] ```token_to_index``` - key,value pair.
   - [x] ```index_to_token``` -key,value pair.
   - [x] ```get_token_for_index``` - returns token for given word index.
   - [x] ```get_index_for_token```  - returns word index for given token(string).
   - [x] ```add_token``` - Add new token to the Vocabulary
   
- prepocessing <br />
   - [x] ```tokenizer``` - can be modified based on need.
   - [x] ```get_context``` - context and word pair.
   - [x] other utility methods.
   
- DatasetNLP <br />
   Converts dataset to Pytorch Dataset subclass.
   
### testMethod.py
  - Test the trained Word2Vec.
  
### main.py <br />
The main file to pass model, parameter, loss function, etc.
