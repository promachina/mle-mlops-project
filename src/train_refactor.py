import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow.tracking.client import MlflowClient

# Function to calculate trip duration in minutes and filter data
def calculate_trip_duration_in_minutes(df):
    df["duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60) & (df['passenger_count'] > 0) & (df['passenger_count'] < 8)]
    return df

def main():
    year = 2021
    month = 1
    color = "green"

    # Download the data
    if not os.path.exists(f"./data/{color}_tripdata_{year}-{month:02d}.parquet"):
        os.system(f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet")

    # Load the data
    df = pd.read_parquet(f"./data/{color}_tripdata_{year}-{month:02d}.parquet")

    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    # Set up the connection to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Setup the MLflow experiment 
    mlflow.set_experiment("green-taxi-monitoring")

    # Preprocess data and select features
    df_processed = calculate_trip_duration_in_minutes(df)
    features = ["PULocationID", "DOLocationID", "trip_distance", "passenger_count", "fare_amount", "total_amount"]
    target = 'duration'
    y = df_processed["duration"]
    X = df_processed.drop(columns=["duration"])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    # Set up Google Cloud credentials
    SA_KEY = os.getenv("SA_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY

    with mlflow.start_run():
        # Log experiment tags
        tags = {
            "model": "linear regression",
            "developer": "<your name>",
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": features,
            "target": target
        }
        mlflow.set_tags(tags)

        # Train the linear regression model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Make predictions and calculate RMSE
        y_pred = lr.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Log the trained model to MLflow
        mlflow.sklearn.log_model(lr, "model")
        run_id = mlflow.active_run().info.run_id

        # Register the model and set it to Production stage
        model_uri = f"runs:/{run_id}/model"
        model_name = "green-taxi-ride-duration"
        mlflow.register_model(model_uri=model_uri, name=model_name)

        model_version = 1
        new_stage = "Production"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=new_stage,
            archive_existing_versions=False
        )

if __name__ == "__main__":
    main()


