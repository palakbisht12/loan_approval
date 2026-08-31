import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

file_name = 'loan.csv'
if not os.path.exists(file_name):
    data = {
        'Applicant_Income': [5000, 3000, 8000, 1500, 4500, 2000, 7000, 2500, 6000, 10000],
        'Loan_Amount': [150, 100, 300, 80, 200, 50, 250, 90, 180, 400],
        'Credit_History': [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
        'Employment_Status': ['Employed', 'Self-Employed', 'Employed', 'Unemployed', 'Employed', 
                             'Unemployed', 'Self-Employed', 'Unemployed', 'Employed', 'Self-Employed'],
        'Married': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
        'Loan_Status': [1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
else:
    df = pd.read_csv(file_name)

encoders = {}
for col in ['Employment_Status', 'Married']:
    if col in df.columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col].astype(str))

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Accuracy: {acc * 100:.2f}%\n")

try:
    income = float(input("Applicant Income: "))
    loan_amt = float(input("Loan Amount: "))
    credit = int(input("Credit History (1/0): "))
    emp = input("Employment Status (Employed/Self-Employed/Unemployed): ").strip()
    married = input("Married (Yes/No): ").strip()

    emp_val = encoders['Employment_Status'].transform([emp])[0]
    married_val = encoders['Married'].transform([married])[0]

    user_sample = pd.DataFrame([[income, loan_amt, credit, emp_val, married_val]], columns=X.columns)
    pred = model.predict(user_sample)[0]

    if pred == 1:
        print("\nDecision: Approved")
    else:
        print("\nDecision: Rejected")

except Exception as e:
    print("Error:", e)

print("\nFeature Importances:")
for col, imp in zip(X.columns, model.feature_importances_):
    print(f"{col}: {imp:.4f}")
