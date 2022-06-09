import pickle
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import os


app = Flask(__name__)


def load_model(filename: str):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def encode(message: str):
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    return model.encode(message)


def predict(reply: str, model) -> str:
    reply_encoded = encode(reply)
    return model.predict([reply_encoded])


@app.route('/', methods=['GET', 'POST'])
def get_prediction():
    reply = request.args.get('reply')
    model = load_model('reply_model.sav')
    reply_type = predict(reply, model)
    return jsonify(reply_type[0])


def main():
    model = load_model('reply_model.sav')
    print(predict('בשמחה אפשר להתחיל טלפונית, יותר קל 5544225552', model))


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
