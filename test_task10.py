import sys
import pickle as p
# import traceback
from flask import Flask, request, jsonify
import json
app = Flask(__name__)

@app.route('/api/summarize', methods=['POST', 'GET'])
def get_url():
    try:
        if request.method == 'POST':
            url = request.json['theUrl']
            #print(content)
            return jsonify(url), 200
    except KeyboardInterrupt:
    print('You cancelled the operation.')
    except NameError:
    print('Not Defined')
    except:
        print("An Error occurred") 

try:
    if __name__ == '__main__':
        app.run(debug=True, host='127.0.0.1', port=5000)
except KeyboardInterrupt:
    print('You cancelled the operation.')
except NameError:
    print('Not Defined')
except:
        print("An exception occurred")
