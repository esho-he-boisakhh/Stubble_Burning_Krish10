import pandas as pd
import joblib
from os import path, getcwd, environ
from google.cloud import aiplatform, storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set up Google Cloud credentials
environ["GOOGLE_APPLICATION_CREDENTIALS"] = path.join(getcwd(), 'credentials.json')

# Configuration data
PROJECT_ID = "smooth-verve-449816-i7"
REGION = "asia-south1"
BUCKET_NAME = "krish_solutionchallenge_bucket-1"
MODEL_NAME = "pollution-predictor"
ENDPOINT_ID = "aiplatform.Endpoint('projects/463687786171/locations/asia-south1/endpoints/3658246709424685056')"
API_KEY = "496755bebf55ba8759e3f2ae51fdd1c4"

#Vertex api
API_KEY_2="AIzaSyBmkISF_47PECjvY45-VSgwoTJXS3HLnZ0"


# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

def load_data():
    city_files = {
        "Amritsar": "Amritsar.csv",
        "Jalandhar": "Jalandhar.csv",
        "Patiala": "Patiala.csv",
        "Ambala": "Ambala.csv",
        "Hisar": "Hisar.csv",
        "Karnal": "Karnal.csv",
        "Panipat": "Panipat.csv"
    }

    df_list = []
    
    for city, file in city_files.items():
        try:
            df = pd.read_csv(path.join(getcwd(), 'Dataset', file))
            df["From Date"] = pd.to_datetime(df["From Date"], format="%d-%m-%Y %H:%M")
            df["To Date"] = pd.to_datetime(df["To Date"], format="%d-%m-%Y %H:%M")
            df["Duration"] = (df["To Date"] - df["From Date"]).dt.total_seconds() / 3600  
            df["City"] = city  # Adding city name as a new column
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")

    if not df_list:
        raise ValueError("No valid data loaded from CSV files.")

    return pd.concat(df_list, ignore_index=True)

def train_and_deploy_model():
    df = load_data().dropna()

    X = df[["Duration", "City"]]
    y = df[["PM2.5", "CO"]]

    # Preprocessing: Scaling + One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Duration"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City"])
        ]
    )

    # Scale target values
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    # Create pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Train-test split & training
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print(f"y_scaler mean: {y_scaler.mean_}, scale: {y_scaler.scale_}")

    # Save model & scalers
    joblib.dump(model, "model.pkl")
    joblib.dump(y_scaler, "y_scaler.pkl")

    # Upload to Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    for file in ["model.pkl", "y_scaler.pkl"]:
        bucket.blob(file).upload_from_filename(file)

    # Deploy to Vertex AI
    try:
        aiplatform.Model.upload(
            display_name=MODEL_NAME,
            artifact_uri=f"gs://{BUCKET_NAME}",
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
        )
        print("✅ Model trained and deployed!")
    except Exception as e:
        print(f"⚠️ Error deploying model: {e}")

if __name__ == "__main__":
    train_and_deploy_model()

