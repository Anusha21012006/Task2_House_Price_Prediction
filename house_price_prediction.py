import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("housing_price_dataset.csv")

print("===== DATASET =====")
print(data.head())

# Convert Neighborhood text into numbers
encoder = LabelEncoder()
data["Neighborhood"] = encoder.fit_transform(data["Neighborhood"])

# Features and Target
X = data[[
    "SquareFeet",
    "Bedrooms",
    "Bathrooms",
    "Neighborhood",
    "YearBuilt"
]]

y = data["Price"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print("Mean Absolute Error:", round(mae, 2))

# Example Prediction
sample_house = [[
    2000,   # SquareFeet
    3,      # Bedrooms
    2,      # Bathrooms
    1,      # Neighborhood
    2000    # YearBuilt
]]

predicted_price = model.predict(sample_house)

print("\nPredicted House Price:")
print(round(predicted_price[0], 2))

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.savefig("prediction_plot.png")

plt.show()

print("\nGraph saved as prediction_plot.png")