# AI-Task-Aventus-Informatics
 The task involves building a **simple Linear Regression model** using Python to analyze the relationship between **hours studied** and **student scores**. The dataset is first loaded and explored, after which a regression model is trained to predict a student’s score based on the number of hours studied. Using the trained model, the score for a student who studies 9 hours is predicted. Finally, the dataset and the fitted regression line are visualized using a plot to demonstrate the model’s performance.
Step 2: Load the dataset
Ensure 'student_scores.csv' is in the same directory
data = pd.read_csv("C:/Users/donsh/Downloads/student_scores.csv")

Display first few rows to verify data
print("Dataset Preview:")
print(data.head())

Step 3: Separate independent (X) and dependent (y) variables
X = data[['Hours']] # Feature (2D array)
y = data['Scores'] # Target variable

Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

Step 5: Predict score for a student studying 9 hours
hours_studied = pd.DataFrame({'Hours': [9]})
predicted_score = model.predict(hours_studied)
print(f"Predicted score for 9 hours studied: {predicted_score[0]:.2f}")

Step 6: Display model parameters
print("\nModel Parameters:")
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (c): {model.intercept_}")

Step 7: Plot data points and regression line
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Hours Studied vs Score")
plt.legend()
plt.show()
