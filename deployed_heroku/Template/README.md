
<div align="center">

![hng](https://res.cloudinary.com/iambeejayayo/image/upload/v1554240066/brand-logo.png)

<br>

</div>

# PAGE SUMMARIZER Model

## INTRODUCTION
This task involves: (1) developing a model that generates the abstractive summary of a Page or an Article.
                                (2) creating an API for a function that receives the url of an article and returns the text in the url.
                     
## GETTING STARTED 
 
### Prerequisite Libraries
- Flask
- Pickle
- Beautiful Soup
- Numpy
- Scikit-Learn
- Networkx
- An Integrated Development Environment(IDE)
- Jupyter Lab

## INSTALLATIONS
- Download and Install the anaconda package for your version of OS here: 
https://www.anaconda.com/distribution/#download-section
- ```pip install Flask``` to install Flask
- ```pip install bs4``` to install the library for web crawling, that is Beautiful Soup

Please note that most of the prerequisite libraries mentioned comes with the Anaconda Package, 
you will only need to import them.

```
import os #os is python's standard library, no need to install it.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle as p
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
```

### CONFIGURATION

### Page Summarizer

```
# First read the file/article text that is to be summarized and perform Data Wrangling on it
file = open("mayowa.txt","r") 
data=file.readlines() 
file.close()
```
```
#Next, define the "word2vec" model to be used
model_w2v = gensim.models.Word2Vec(
            tokenized_text,
            size=200, # desired no. of features/independent variables 
            window=5, # context window size
            min_count=2,
            sg = 0, # 1 for cbow model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)

```

```
# train the model and compute the similarity score using cosine similarity matrix
sim_mat = np.zeros([len(wordvec_arrays), len(wordvec_arrays)]
for i in range(len(wordvec_arrays)):
  for j in range(len(wordvec_arrays)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(wordvec_arrays[i].reshape(1,200), wordvec_arrays[j].reshape(1,200))[0,0]
```

```
# Generate a graph using Networkx, compute the pagerank scores and sort them 
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
sorted_x = sorted(scores.items(), key=lambda kv: kv[1],reverse=True)

sent_list=[]
for i in sorted_x:
    sent_list.append(i[0])
```

### Web Scraping

```
# Scraping the text content from the url of a sample article from wikipedia

def read_content():  
    url = "https://en.wikipedia.org/wiki/Machine_learning"
    response = requests.get(url)
    html = response.content
    soup = BeautifulSoup(html)
    text = soup.text
    return text

print(read_content())
```

```
# Creating an API for the text content in json format
def get_url():
    if request.method == 'POST':
        url = request.json['theUrl']
        #print(content)
        jsonified = jsonify(url), 200
        return jsonified
```
### OUTPUT
```
#To extract a summarized text in 15 sentences
num=15
summary=''
for i in range(num):
    summary=summary+all_sentences[sent_list[i]]+'. '
print(summary)
```

## CONCLUSION
To summarize the article, please make a POST request to the /api/summarize endpoint, use the format {"theUrl": "the URL of the article"}"""}

### Hosted on heroku @ https://summarizerheroku.herokuapp.com and on git @ 
https://git.heroku.com/summarizerheroku.git 
