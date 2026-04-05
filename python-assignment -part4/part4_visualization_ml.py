import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# loading dataset
df = pd.read_csv("students.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nSummary stats:")
print(df.describe())

print("\nPass/Fail count:")
print(df['passed'].value_counts())

# subject columns
subject_cols = ['math', 'science', 'english', 'history', 'pe']

# average scores
print("\nAverage scores (Pass students):")
print(df[df['passed'] == 1][subject_cols].mean())

print("\nAverage scores (Fail students):")
print(df[df['passed'] == 0][subject_cols].mean())

# highest average student
df['avg_score'] = df[subject_cols].mean(axis=1)

top_student = df.loc[df['avg_score'].idxmax()]
print("\nTop student:", top_student['name'], "Avg:", top_student['avg_score'])
avg_scores = df[subject_cols].mean()

plt.figure()
plt.bar(avg_scores.index, avg_scores.values)
plt.title("Average Score per Subject")
plt.xlabel("Subjects")
plt.ylabel("Average Score")
plt.savefig("plot1_bar.png")
plt.show()
plt.figure()
plt.hist(df['math'], bins=5)
mean_val = df['math'].mean()

plt.axvline(mean_val, linestyle='--')
plt.title("Math Score Distribution")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.savefig("plot2_hist.png")
plt.show()
plt.figure()

pass_data = df[df['passed'] == 1]
fail_data = df[df['passed'] == 0]

plt.scatter(pass_data['study_hours_per_day'], pass_data['avg_score'], label="Pass")
plt.scatter(fail_data['study_hours_per_day'], fail_data['avg_score'], label="Fail")

plt.xlabel("Study Hours")
plt.ylabel("Average Score")
plt.title("Study Hours vs Avg Score")
plt.legend()

plt.savefig("plot3_scatter.png")
plt.show()
pass_att = df[df['passed']==1]['attendance_pct']
fail_att = df[df['passed']==0]['attendance_pct']

plt.figure()
plt.boxplot([pass_att, fail_att], labels=["Pass", "Fail"])

plt.title("Attendance Comparison")
plt.ylabel("Attendance %")

plt.savefig("plot4_box.png")
plt.show()
plt.figure()

plt.plot(df['name'], df['math'], marker='o', label="Math")
plt.plot(df['name'], df['science'], marker='x', label="Science")

plt.xticks(rotation=45)
plt.xlabel("Students")
plt.ylabel("Score")
plt.title("Math vs Science Scores")
plt.legend()

plt.savefig("plot5_line.png")
plt.show()
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.barplot(data=df, x='passed', y='math')
plt.title("Math vs Pass")

plt.subplot(1,2,2)
sns.barplot(data=df, x='passed', y='science')
plt.title("Science vs Pass")

plt.savefig("plot6_seaborn_bar.png")
plt.show()
plt.figure()

sns.scatterplot(data=df, x='attendance_pct', y='avg_score', hue='passed')

sns.regplot(data=df[df['passed']==1], x='attendance_pct', y='avg_score')
sns.regplot(data=df[df['passed']==0], x='attendance_pct', y='avg_score')

plt.title("Attendance vs Avg Score")

plt.savefig("plot7_seaborn_scatter.png")
plt.show()
# I found seaborn easier for styling and quick plots like barplot and scatterplot.
# Compared to matplotlib, seaborn needs less code for better visuals.
# But matplotlib gives more control for custom plots like line and box plots.
X = df[['math','science','english','history','pe','attendance_pct','study_hours_per_day']]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
print("Training Accuracy:", train_acc)
y_pred = model.predict(X_test_scaled)

test_acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_acc)

# show predictions
names = df.loc[X_test.index, 'name']

for i in range(len(names)):
    result = "✅" if y_pred[i] == y_test.iloc[i] else "❌"
    print(names.iloc[i], "| Actual:", y_test.iloc[i], "| Pred:", y_pred[i], result)
    
features = X.columns
coeff = model.coef_[0]

feature_imp = sorted(zip(features, coeff), key=lambda x: abs(x[1]), reverse=True)

print("\nFeature Importance:")
for f, c in feature_imp:
    print(f, ":", c)

# plot
values = [c for f,c in feature_imp]
labels = [f for f,c in feature_imp]# ============================================
# Student Performance Analysis & Prediction
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# TASK 1 - DATA EXPLORATION
# -------------------------------

# loading dataset
df = pd.read_csv("students.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nSummary stats:")
print(df.describe())

print("\nPass/Fail count:")
print(df['passed'].value_counts())

# subject columns
subject_cols = ['math', 'science', 'english', 'history', 'pe']

print("\nAverage scores (Pass students):")
print(df[df['passed'] == 1][subject_cols].mean())

print("\nAverage scores (Fail students):")
print(df[df['passed'] == 0][subject_cols].mean())

# adding avg score column
df['avg_score'] = df[subject_cols].mean(axis=1)

top_student = df.loc[df['avg_score'].idxmax()]
print("\nTop student:", top_student['name'], "| Avg:", top_student['avg_score'])

# -------------------------------
# TASK 2 - MATPLOTLIB PLOTS
# -------------------------------

# 1. Bar Chart
avg_scores = df[subject_cols].mean()

plt.figure()
plt.bar(avg_scores.index, avg_scores.values)
plt.title("Average Score per Subject")
plt.xlabel("Subjects")
plt.ylabel("Average Score")
plt.savefig("plot1_bar.png")
plt.show()

# 2. Histogram
plt.figure()
plt.hist(df['math'], bins=5)

mean_val = df['math'].mean()
plt.axvline(mean_val, linestyle='--', label='Mean')

plt.title("Math Score Distribution")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("plot2_hist.png")
plt.show()

# 3. Scatter Plot
plt.figure()

pass_data = df[df['passed'] == 1]
fail_data = df[df['passed'] == 0]

plt.scatter(pass_data['study_hours_per_day'], pass_data['avg_score'], label="Pass")
plt.scatter(fail_data['study_hours_per_day'], fail_data['avg_score'], label="Fail")

plt.xlabel("Study Hours")
plt.ylabel("Average Score")
plt.title("Study Hours vs Avg Score")
plt.legend()
plt.savefig("plot3_scatter.png")
plt.show()

# 4. Box Plot
pass_att = df[df['passed']==1]['attendance_pct']
fail_att = df[df['passed']==0]['attendance_pct']

plt.figure()
plt.boxplot([pass_att, fail_att], labels=["Pass", "Fail"])
plt.title("Attendance Comparison")
plt.ylabel("Attendance %")
plt.savefig("plot4_box.png")
plt.show()

# 5. Line Plot
plt.figure()

plt.plot(df['name'], df['math'], marker='o', label="Math")
plt.plot(df['name'], df['science'], marker='x', label="Science")

plt.xticks(rotation=45)
plt.xlabel("Students")
plt.ylabel("Score")
plt.title("Math vs Science Scores")
plt.legend()
plt.savefig("plot5_line.png")
plt.show()

# -------------------------------
# TASK 3 - SEABORN
# -------------------------------

# seaborn bar plots
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.barplot(data=df, x='passed', y='math')
plt.title("Math vs Pass")

plt.subplot(1,2,2)
sns.barplot(data=df, x='passed', y='science')
plt.title("Science vs Pass")

plt.savefig("plot6_seaborn_bar.png")
plt.show()

# seaborn scatter + regression
plt.figure()

sns.scatterplot(data=df, x='attendance_pct', y='avg_score', hue='passed')

sns.regplot(data=df[df['passed']==1], x='attendance_pct', y='avg_score', label='Pass')
sns.regplot(data=df[df['passed']==0], x='attendance_pct', y='avg_score', label='Fail')

plt.title("Attendance vs Avg Score")
plt.legend()
plt.savefig("plot7_seaborn_scatter.png")
plt.show()

# My observation:
# seaborn is easier for quick plots and better visuals
# matplotlib gives more control for customizing graphs

# -------------------------------
# TASK 4 - MACHINE LEARNING
# -------------------------------

# preparing data
X = df[['math','science','english','history','pe','attendance_pct','study_hours_per_day']]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
print("\nTraining Accuracy:", round(train_acc, 2))

# testing
y_pred = model.predict(X_test_scaled)

test_acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", round(test_acc, 2))

# prediction comparison
names = df.loc[X_test.index, 'name']

print("\nPredictions:")
for i in range(len(names)):
    result = "✅" if y_pred[i] == y_test.iloc[i] else "❌"
    print(names.iloc[i], "| Actual:", y_test.iloc[i], "| Pred:", y_pred[i], result)

# -------------------------------
# Feature Importance
# -------------------------------

coeff = model.coef_[0]
features = X.columns

feature_imp = sorted(zip(features, coeff), key=lambda x: abs(x[1]), reverse=True)

print("\nFeature Importance:")
for f, c in feature_imp:
    print(f, ":", round(c, 3))

# plot importance
values = [c for f,c in feature_imp]
labels = [f for f,c in feature_imp]

colors = ['green' if v > 0 else 'red' for v in values]

plt.figure()
plt.barh(labels, values, color=colors)
plt.title("Feature Importance")
plt.xlabel("Coefficient")
plt.ylabel("Features")
plt.savefig("plot8_feature_importance.png")
plt.show()

# -------------------------------
# Bonus - New Student Prediction
# -------------------------------

new_student = [[75, 70, 68, 65, 80, 82, 3.2]]

new_scaled = scaler.transform(new_student)

pred = model.predict(new_scaled)[0]
prob = model.predict_proba(new_scaled)

print("\nNew Student Prediction:")
print("Pass" if pred == 1 else "Fail")
print("Probability:", prob)

colors = ['green' if v > 0 else 'red' for v in values]

plt.figure()
plt.barh(labels, values)
plt.title("Feature Importance")
plt.xlabel("Coefficient")
plt.ylabel("Features")

plt.savefig("plot8_feature_importance.png")
plt.show()
new_student = [[75, 70, 68, 65, 80, 82, 3.2]]

new_scaled = scaler.transform(new_student)

pred = model.predict(new_scaled)[0]
prob = model.predict_proba(new_scaled)

print("\nNew Student Prediction:")
print("Pass" if pred == 1 else "Fail")
print("Probability:", prob)