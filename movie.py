#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# In[2]:


df=pd.read_csv("movie.csv")

df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:


df.duplicated().any()


# In[8]:


df.drop('Poster_Link',axis=1,inplace=True)
df.drop('Star1',axis=1,inplace=True)
df.drop('Star2',axis=1,inplace=True)
df.drop('Star3',axis=1,inplace=True)
df.drop('Star4',axis=1,inplace=True)
df.drop('Certificate',axis=1,inplace=True)
df.head()


# In[9]:


df.columns


# In[10]:


df['Genre'].unique()


# In[11]:


df = df.loc[df['Genre']!='unknown']
df.reset_index(drop = True, inplace = True)


# In[12]:


genres=pd.value_counts(df.Genre)
print('There are ',len(genres), 'different Genres in the dataset:')
print('-'*50)
print(genres)


# In[13]:


top_genre = pd.DataFrame(genres[:11]).reset_index()
top_genre.columns = ['genres', 'number_of_movies']
top_genre


# In[14]:


conditions = [df['Genre']=='Drama',df['Genre']=='Drama, Romance',df['Genre']=='Comedy, Drama', df['Genre']=='Comedy, Drama, Romance', 
              df['Genre']=='Action, Crime, Drama',df['Genre']=='Biography, Drama, History', df['Genre']=='Crime, Drama, Thriller',
              df['Genre']=='Crime, Drama, Mystery',df['Genre']==' Crime, Drama',df['Genre']=='Animation, Adventure, Comedy', df['Genre']=='Action, Adventure, Sci-Fi',]
choices = [1,2,3,4,5,6,7,8,9,10,11]
df['labels'] = np.select(conditions, choices, 0)


# In[15]:


df.sample(3)


# In[16]:


df['labels'].value_counts()


# In[17]:


df_to = (df.loc[df['labels']!=0]).reset_index(drop = True)


# In[18]:


df_to.head()


# In[19]:


df_to.drop(columns = ['Released_Year', 'Runtime', 'Director', 'Meta_score', 'No_of_Votes','Gross'], axis = 1, inplace = True)
df_to.head(3)


# In[20]:


print(df["Overview"][0])


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(df['Overview'], df['Genre'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[22]:


print(clf.predict(count_vect.transform(["Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."])))


# In[ ]:




