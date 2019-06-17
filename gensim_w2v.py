from gensim.models import word2vec
from gensim.models import Word2Vec
import numpy as np
import os
from random import shuffle
import re
import urllib.request
import zipfile
import lxml.etree


# how to perform word embedding with Gensim, a powerful NLP toolkit, and a TED Talk dataset

# 1-- download the the dataset using urllib
url = 'https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip'
urllib.request.urlretrieve(url, filename="ted_en-20160408.zip")

# 2-- extracting the subtitle from the file
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

#print(doc)

input_text = '\n'.join(doc.xpath('//content/text()')) # ----> take the content of the site
#print(input_text)

"""
Clearly, there are some redundant information that is not helpful for us to understand the talk, 
such as the words describing sound in the parenthesis and the speaker’s name. 
We get rid of these words with regular expression.
"""
# remove parenthesis: dau ngoac don
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
#print(input_text_noparens)

# store as list of sentences
sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    #print(line)
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    #rint(m.groupdict()['postcolon'])
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
#print(sentences_strings_ted)

# store as list of lists of words
sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)

#print(sentences_ted)

model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=1)

"""
    sentences: the list of split sentences.
    size: the dimensionality of the embedding vector
    window: the number of context words you are looking at
    min_count: tells the model to ignore words with total count less than this number.
    workers: the number of threads being used
    sg: whether to use skip-gram (0) or CBOW (1)
"""

#guess = model_ted.wv.most_similar('girl')
#for w in guess:
 #   print(w)

# FastText
"""
FastText is an extension to Word2Vec proposed by Facebook in 2016. 
Instead of feeding individual words into the Neural Network, 
FastText breaks words into several n-grams (sub-words). 
For instance, the tri-grams for the word apple is app, ppl, and ple 
(ignoring the starting and ending of boundaries of words). 
The word embedding vector for apple will be the sum of all these n-grams. 
After training the Neural Network, we will have word embeddings for all the n-grams 
given the training dataset. Rare words can now be properly represented 
since it is highly likely that some of their n-grams also appears in other words. 

"""
from gensim.models import FastText

model_FastText = FastText(sentences_ted, size=100, window=5, min_count=5, workers=4,sg=1)
no_fast = model_ted.wv.most_similar('man')
fast = model_FastText.wv.most_similar('man')
for i in no_fast:
    print(i)

print("-----------------------------------------")

for j in fast:
    print(j)
