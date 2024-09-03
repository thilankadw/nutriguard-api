from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('obesity_model.pkl')

@app.route("/")
def home():
    return "Home"

@app.route('/predict-obesity', methods=['POST'])
def predict():
    try:
        data = request.json

        calory = data['features'].get('calory')
        fat = data['features'].get('fat')
        sugar = data['features'].get('sugar')

        model = joblib.load('obesity_model.pkl')
        
        single_sample = pd.DataFrame({
            'calory': [calory],
            'fat': [fat],
            'sugar': [sugar]
        })

        print(single_sample)

        single_sample_array = single_sample.values

        prediction = model.predict(single_sample_array)

        prediction = prediction[0].item()

        return jsonify({'prediction': prediction})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        return jsonify({'error': 'An error occurred: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
