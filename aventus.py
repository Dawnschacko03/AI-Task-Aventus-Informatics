# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Load the dataset
# Ensure 'student_scores.csv' is in the same directory
data = pd.read_csv("C:/Users/donsh/Downloads/student_scores.csv")

# Display first few rows to verify data
print("Dataset Preview:")
print(data.head())

# Step 3: Separate independent (X) and dependent (y) variables
X = data[['Hours']]   # Feature (2D array)
y = data['Scores']             # Target variable

# Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 5: Predict score for a student studying 9 hours
hours_studied = pd.DataFrame({'Hours': [9]})
predicted_score = model.predict(hours_studied)

print(f"Predicted score for 9 hours studied: {predicted_score[0]:.2f}")


# Step 6: Display model parameters
print("\nModel Parameters:")
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (c): {model.intercept_}")

# Step 7: Plot data points and regression line
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Hours Studied vs Score")
plt.legend()
plt.show()