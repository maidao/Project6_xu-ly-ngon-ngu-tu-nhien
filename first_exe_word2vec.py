import math
import numpy as np
from gensim.models import word2vec

#Tính sigmoid probability

"""
z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
z_exp = [math.exp(i) for i in z]
print([round(i, 2) for i in z_exp])
sum_z_exp = sum(z_exp)
print(round(sum_z_exp, 2))
softmax = [i / sum_z_exp for i in z_exp]
print([round(i, 3) for i in softmax])
"""

#Tính sigmoid probability voi numpy

z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
softmax = np.exp(z)/np.sum(np.exp(z))
print(softmax)

# Dự đoán probability của từ:
num_features = 300  # Word vector dimensionality
min_word_count = 2  # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
#downsampling = 1e-3 # (0.001) Downsample setting for frequent words

# Initializing the train model
sentences = 'I ,hate, bananas'
line = sentences.split(',')
for i in line:
    print(i)
print("Training model....", len(sentences))
model = word2vec.Word2Vec(line,\
                          workers = num_workers,\
                          size = num_features,\
                          min_count = min_word_count,\
                          window = context)

# To make the model memory efficient
#model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
#model_name = "300features_40minwords_10context"
#model.save(model_name)
g = model.wv.most_similar('bananas')
for i in g:
    print(i)

