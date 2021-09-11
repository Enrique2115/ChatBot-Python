# install libraries Flask, numpy, nltk, tensorflow, pickle5, pickle-mixin, PyYAML

import json
import pickle
import random
import nltk
import numpy
from nltk.stem import LancasterStemmer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import model_from_yaml

nltk.download('punkt')

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("chatbot.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for patterns in intent["patterns"]:
            wrds = nltk.word_tokenize(patterns)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("chatbot.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)

try:
    yaml_file = open('chatbotmodel.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    myChatModel = model_from_yaml(loaded_model_yaml)
    myChatModel.load_weights("chatbotmodel.h5")
    print("Modelo cargado desde disco")

except:
    # Realizando nuestra red neuronal
    myChatModel = Sequential()
    myChatModel.add(Dense(8, input_shape=[len(words)], activation='relu'))
    myChatModel.add(Dense(len(labels), activation='softmax'))

    # Optimizar el Modelo
    myChatModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entrenando el Modelo
    myChatModel.fit(training, output, epochs=1000, batch_size=8)

    # Serializando el Modelo yaml y guardando en el disco
    model_yaml = myChatModel.to_yaml()
    with open("chatbotmodel.yaml", "w") as y_file:
        y_file.write(model_yaml)

    # serializar pesos a HDF5
    myChatModel.save_weights("chatbotmodel.h5")
    print("Modelo Guardado en el disco")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chatWithBot(inputText):
    currentText = bag_of_words(inputText, words)
    currentTextArray = [currentText]
    numpyCurrentText = numpy.array(currentTextArray)

    if numpy.all((numpyCurrentText == 0)):
        return "No lo entendí, inténtalo de nuevo"

    result = myChatModel.predict(numpyCurrentText[0:1])
    result_index = numpy.argmax(result)
    tag = labels[result_index]

    if result[0][result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses)

    else:
        return "No lo entendí, inténtalo de nuevo"

def chat():
    print("Empiece a hablar con el chatbot (Presione quit para Salir)")

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        print(chatWithBot(inp))

#chat()
