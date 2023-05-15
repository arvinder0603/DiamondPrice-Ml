from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

def load_model():
    # Load your trained machine learning model here
    model = joblib.load('models.pkl')
    return model

def predict_diamond_price(features):
    # Load the trained model
    model = load_model()

    # Preprocess the input features
    features = np.array(features).reshape(1, -1)

    # Make the prediction
    price = model.predict(features)

    return price[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        carat = float(request.form['carat'])
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])

        # Create a list of features
        features = [carat, cut, color, clarity, depth, table, x, y, z]

        # Predict the diamond price
        price = predict_diamond_price(features)

        return render_template('results.html', features=features, price=price)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
