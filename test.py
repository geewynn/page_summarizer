import pickle as p
# import traceback
from flask import Flask, request, jsonify
import json
app = Flask(__name__)

#@app.before_request
#def exe():
#    summarizer = 'modelfile.pkl'
#    model = p.load(open(summarizer, 'rb'))


@app.route('/api/summarize', methods=['POST', 'GET'])
def test():
    if request.method == 'POST':
        content = request.json()
        print(content)
        return jsonify(content), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

