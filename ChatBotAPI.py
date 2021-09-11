#importamos librerias necesarias
from flask import Flask, request, jsonify

#importamos la funcion de la clase main
from main import chatWithBot

app = Flask(__name__)

#creamos la ruta de la api
@app.route('/chat', methods =['GET', 'POST'])
def chatBot():
    chatInput = request.form['chatInput']
    return jsonify(chatBotReply=chatWithBot(chatInput))


if __name__ == '__main__':
    app.run()