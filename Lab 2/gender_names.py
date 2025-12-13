#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[3]:


nltk.download('names')


# In[12]:


def gender_features(word):
    return {"last_letter": word[-1]}

gender_features('obama')


# In[5]:


from nltk.corpus import names
names.words()
print(len(names.words()))


# In[8]:


labeled_names=([(name,'male') for name in names.words('male.txt')] +[(name,'female') for name in names.words('female.txt')])


# In[18]:


import random
random.shuffle(labeled_names)
import nltk
featuresets=[(gender_features(n),gender) for (n,gender) in labeled_names]
train_set, test_set=featuresets[5000:],featuresets[:2000]
classifier=nltk.NavieBayesClassifier.train(train_set)
classifier.classify(gender_features("David"))
classifier.classify(gender_features("Yann Lecun"))
print(nltk.classify.accuracy(classifier,test_test))


# In[ ]:




