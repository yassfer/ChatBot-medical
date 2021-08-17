import nltk
from nltk.stem.lancaster import LancasterStemmer #used to stem our words
stemmer=LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import nltk
import pickle
nltk.download('punkt')

from flask import Flask, jsonify, request

main = Flask(__name__)

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except :
    words=[]
    labels=[]
    doc1=[]
    doc2=[]

    for intent in data["intents"] :
        for pattern in intent["patterns"] :
            wrds=nltk.word_tokenize(pattern)
            words.extend(wrds)
            doc1.append(wrds)
            doc2.append(intent["tag"])
            if intent["tag"] not in labels :
                labels.append(intent["tag"])
    words=[stemmer.stem(w.lower() ) for w in words ]
    words= sorted(list(set(words)))
    labels=sorted(labels)

    #neural networks only understand numbers
    #use bag of words using  hot encoding

    training=[]
    output=[]
    out_empty=[0 for _ in range(len(labels))]

    for x , doc in enumerate(doc1) :
        bag=[]
        wrds=[stemmer.stem(w) for w in doc if w not in "?"]
        for w in words :
            if w in wrds :
                bag.append(1)
            else :
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(doc2[x])] = 1

        training.append(bag)
        output.append(output_row)

    training=numpy.array(training)
    output=numpy.array(output)

with open("data.pickle" , "wb") as f :
    pickle.dump((words , labels , training , output) , f)
from tensorflow.python.framework import ops
ops.reset_default_graph()


net= tflearn.input_data(shape=[None , len(training[0])])
net=tflearn.fully_connected(net , 8 ) #8 neurons
net=tflearn.fully_connected(net , 8 )
net=tflearn.fully_connected(net , len(output[0]) , activation="softmax")
net=tflearn.regression(net)

model=tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training , output , n_epoch=1000 , batch_size= 8 , show_metric=True)
    model.save("model.tflearn")


#predictions

def bag_of_words(s , words) :
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]

    for se in s_words :
        for i , w in enumerate(words) :
            if w == se : #current word is equal in the word in our sentence
                bag[i] =1
    return numpy.array(bag)

def chat() :
    print("Start talking with the bot , type quit to stop the chatbot")
    while True :
        inp=input("you : ")
        if inp.lower() == "quit" :
            break
        results= model.predict([bag_of_words(inp , words)] )
        results_index=numpy.argmax(results)
        tag=labels[results_index]
        print(tag)
        for tg in data["intents"] :
            if tg["tag"]==tag :
                responses=tg["responses"]
                print(random.choice(responses))
                

@main.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response

@main.route("/chat/<string:inp>", methods=['GET'])
def chatDef(inp):
    print("Start talking with the bot , type quit to stop the chatbot")
    while True :
        #inp=input("you : ")
        if inp.lower() == "quit" :
            break
        results= model.predict([bag_of_words(inp , words)] )
        results_index=numpy.argmax(results)
        tag=labels[results_index]
        print(tag)
        for tg in data["intents"] :
            if tg["tag"]==tag :
                responses=tg["responses"]
                #print(random.choice(responses))
                #return random.choice(responses)
                return jsonify({"responses": random.choice(responses)})
if __name__ == "__main__":
    main.run()