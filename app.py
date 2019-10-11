""" This script loads in a model, gets a Url, gets an article in the URL and summarizes the article """

import pickle as p
import traceback
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# loading the model
summarizer = 'model.pkl'
model = p.load(open(summarizer, 'rb'))

app = Flask(__name__)
CORS(app)


@app.route('/api/summarize', methods=['POST', 'GET'])
def summarize():
    """ Returns summary of articles """
    if model:
        try:
            if request.method == 'POST':
                url = request.json['theUrl']
                requested_url = urllib.request.urlopen(url).read().decode('utf8','ignore')
                soup= BeautifulSoup(requested_url, 'html.parser')
                return soup

            # function to target only tag <p>
            def find_all_text():
                text_p = read_content().find_all('p')
                text =''
                for i in range(0,len(text_p)):
                    text += text_p[i].text
                    text = text.lower()
                # tokenize the text
                tokens =[t for t in text.split()]
                clean_token =tokens[:]
                #define irrelevant words that include stop words , punctuations and numbers
                stopword = set(stopwords.words('english') + list(punctuation))
                for token in tokens:
                    if token in stopword:
                        clean_token.remove(token)
                freq = nltk.FreqDist(clean_token)
                top_words=[]
                top_words=freq.most_common(100)
                sentences = sent_tokenize(text)
                #Iterating through all the sentences from the web to create a ranking for each sentence

                ranking = defaultdict(int)
                for i, sent in enumerate(sentences):
                    for word in word_tokenize(sent.lower()):
                        if word in freq:
                            ranking[i]+=freq[word]
                    top_sentences = nlargest(10, ranking, ranking.get)
                #Sorting all sentences
                sorted_sentences = [sentences[j] for j in sorted(top_sentences)]
                print(sorted_sentences)
                with open('sorted_sentences.txt', 'w') as f:
                    for i in sorted_sentences:
                        f.write(i+"\n")


            #print (find_all_text())
                summary = model.summarize(article)
                return jsonify(summary=summary)
            return {'data': """Welcome to the Page Summarizer API.
            To summarize please make a POST request to the /api/summarize endpoint
            using the format {"theUrl": "the URL of the article"}"""}
        except Exception:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('You need a trained model first')
        return 'Model not found'

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
