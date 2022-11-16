from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from joblib import load
import sklearn


#curl -d "{\"Medidas\":[[1,2,3,4]]}" -H "Content-Type: application/json" -X POST http://127.0.0.1:5000/predecir

app= Flask(__name__)

CORS(app)

@app.route("/")
def home():
    return 'La pagina esta funcionando bien'

@cross_origin
@app.route("/predecir", methods=["POST"])
def predecir():

    json=request.get_json(force=True)
    medidas=json['Medidas']

    clf = load('fireforestDetect.pkl')
    prediccion=clf.predict(medidas)

    return str(prediccion)


if __name__ == '__main__':
    app.run()