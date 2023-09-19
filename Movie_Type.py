import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic movie dataset
np.random.seed(0)
n_samples = 1000

data = {
    'Genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror'], size=n_samples),
    'Director': np.random.choice(['Director A', 'Director B', 'Director C'], size=n_samples),
    'Actor_1': np.random.choice(['Actor X', 'Actor Y', 'Actor Z'], size=n_samples),
    'Actor_2': np.random.choice(['Actor X', 'Actor Y', 'Actor Z'], size=n_samples),
    'Budget': np.random.randint(1000000, 10000000, size=n_samples),
    'Rating': np.random.uniform(1, 10, size=n_samples)
}

df = pd.DataFrame(data)

# Encode categorical features
df = pd.get_dummies(df, columns=['Genre', 'Director', 'Actor_1', 'Actor_2'], drop_first=True)

# Define features and target variable
X = df.drop('Rating', axis=1)
y = df['Rating']

# Split the data into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Create a scatter plot of actual vs. predicted ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs. Predicted Movie Ratings")

# Show the plot
plt.show()
