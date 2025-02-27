import requests
import joblib
import pandas as pd
from flask import Flask, jsonify

API_KEY = "496755bebf55ba8759e3f2ae51fdd1c4"

CITIES = {
    "Amritsar": (31.6340, 74.8723),
    "Jalandhar": (31.3260, 75.5762),
    "Patiala": (30.3395, 76.3862),
    "Ambala": (30.3780, 77.0400),
    "Hisar": (29.1490, 75.7227),
    "Karnal": (29.6840, 76.9834),
    "Panipat": (29.3918, 76.9696)
}

app = Flask(__name__)

def get_real_time_pollution(lat, lon):
    """Fetch real-time pollution data from OpenWeather API"""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["list"][0]["components"] if "list" in data and data["list"] else None
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching pollution data: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    """Predict pollution levels and detect stubble burning"""
    results = []
    try:
        model = joblib.load("model.pkl")
        y_scaler = joblib.load("y_scaler.pkl")
        preprocessor = model.named_steps["preprocessor"]

        for city, (lat, lon) in CITIES.items():
            pollution_data = get_real_time_pollution(lat, lon)
            if not pollution_data:
                results.append({"city": city, "error": "Could not fetch real-time pollution data."})
                continue
            
            pollution_data["co"] = pollution_data["co"] / 1000  
            
            # Prepare input data
            input_data = pd.DataFrame([[24, city]], columns=["Duration", "City"])
            transformed_data = preprocessor.transform(input_data)

            # Predict pollution levels
            scaled_prediction = model.named_steps["regressor"].predict(transformed_data)[0]

            # Manually inverse transform predictions
            pm25_mean, co_mean = y_scaler.mean_
            pm25_scale, co_scale = y_scaler.scale_

            predicted_pm2_5 = (scaled_prediction[0] * pm25_scale) + pm25_mean
            predicted_co = (scaled_prediction[1] * co_scale) + co_mean  

            # Debugging
            print(f"Predicted Scaled CO: {scaled_prediction[1]}, Fixed CO: {predicted_co}")

            # Detect stubble burning
            stubble_burning_detected = (
                (pollution_data["pm2_5"] > predicted_pm2_5 * 2) or 
                (pollution_data["co"] > predicted_co * 2)
            )


            results.append({
            "city": city,
            "predicted_pm2_5": round(predicted_pm2_5, 2),
            "predicted_co": round(predicted_co, 2),
            "real_pm2_5": pollution_data["pm2_5"],
            "real_co": pollution_data["co"],
            "stubble_burning_detected": bool(stubble_burning_detected)
            })
        
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
