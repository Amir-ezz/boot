from flask import Flask, request, jsonify
import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input

app = Flask(__name__)

# تحميل البيانات
with open('intents.json') as file:
    data = json.load(file)

# تحميل النموذج
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# تعريف نقطة النهاية للتنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # احصل على البيانات المرسلة في الطلب
    user_input = data['user_input']  # افحص البيانات المرسلة للتنبؤ

    # القيام بعملية التنبؤ باستخدام النموذج
    results = model.predict(np.array([bag_of_words(user_input, words)]))
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    # استعد الرد لإرساله كإجابة JSON
    response = {"response": random.choice(responses)}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
