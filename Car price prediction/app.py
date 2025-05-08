from flask import Flask, render_template, request
import joblib
import numpy as np
import webbrowser  # Importing the webbrowser module

app = Flask(__name__)

# ✅ Correct file path using raw string
model = joblib.load(r'C:\Users\nakul\OneDrive\Desktop\Car price prediction\saved models\car_price_model.sav')

# Define the exact feature list you used while training
columns = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
           'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm',
           'citympg', 'highwaympg',
           'fueltype_gas', 'aspiration_turbo', 'doornumber_two',
           'carbody_hardtop', 'carbody_hatchback', 'carbody_sedan', 'carbody_wagon',
           'drivewheel_fwd', 'drivewheel_rwd']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numeric features
        numeric_data = [
            float(request.form['symboling']),
            float(request.form['wheelbase']),
            float(request.form['carlength']),
            float(request.form['carwidth']),
            float(request.form['carheight']),
            float(request.form['curbweight']),
            float(request.form['enginesize']),
            float(request.form['boreratio']),
            float(request.form['stroke']),
            float(request.form['compressionratio']),
            float(request.form['horsepower']),
            float(request.form['peakrpm']),
            float(request.form['citympg']),
            float(request.form['highwaympg']),
        ]

        # One-hot encoded categorical features
        fueltype = request.form['fueltype']
        aspiration = request.form['aspiration']
        doornumber = request.form['doornumber']
        carbody = request.form['carbody']
        drivewheel = request.form['drivewheel']

        categorical_data = [
            1 if fueltype == 'gas' else 0,
            1 if aspiration == 'turbo' else 0,
            1 if doornumber == 'two' else 0,
            1 if carbody == 'hardtop' else 0,
            1 if carbody == 'hatchback' else 0,
            1 if carbody == 'sedan' else 0,
            1 if carbody == 'wagon' else 0,
            1 if drivewheel == 'fwd' else 0,
            1 if drivewheel == 'rwd' else 0,
        ]

        final_input = np.array(numeric_data + categorical_data).reshape(1, -1)

        # Prediction
        prediction = model.predict(final_input)[0]
        return render_template('index.html', prediction_text=f'Predicted Car Price: ${prediction:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    # Automatically open the browser when the app starts
    webbrowser.open("http://127.0.0.1:5000")  # Opens the browser
    app.run(debug=True)
