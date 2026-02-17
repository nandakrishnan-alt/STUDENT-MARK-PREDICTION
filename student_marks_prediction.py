import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# 1 loadset
data = pd.read_csv("student_marks.csv")
# 2️ Features & Target
X = data[['Hours']]
y = data['Marks']
# 3️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 4️ Train Model
model = LinearRegression()
model.fit(X_train, y_train)
# 5️ Predictions
predictions = model.predict(X_test)
# 6️ Evaluation
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)
# 7️ Predict Custom Input
hours =float(input("Enter the number of hours: "))
predicted_marks = model.predict(pd.DataFrame([[hours]], columns=['Hours']))
print("Predicted Marks:", predicted_marks[0])
# 8️ Visualization (Optional, commented out for non-interactive run)
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()
