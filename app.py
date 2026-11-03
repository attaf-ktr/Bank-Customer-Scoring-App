from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Le modèle est chargé ici en mémoire une seule fois au démarrage
# with open('notebooks_and_eda/votre_modele.pkl', 'rb') as f:
#     model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # prediction = model.predict([data['features']])
    # return jsonify({'prediction': int(prediction[0])})
    return jsonify({"status": "API Active", "message": "Endpoint de prediction pret."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
