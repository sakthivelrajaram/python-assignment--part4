# 📊 Student Performance Analysis & Prediction

## 📌 Overview
This project analyzes a small student dataset, creates visualizations, and builds a machine learning model to predict whether a student will pass or fail.

It covers the full workflow: data loading, exploration, visualization, and prediction.

---

## 🗂️ Dataset
The dataset used is `students.csv`.

It contains:
- Student name
- Subject scores: Math, Science, English, History, PE
- Attendance percentage
- Study hours per day
- Pass/Fail (1 = Pass, 0 = Fail)

---

## ⚙️ Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📊 Tasks Implemented

### 🔹 Task 1: Data Exploration
- Displayed first 5 rows
- Checked dataset shape and data types
- Generated summary statistics
- Counted pass and fail students
- Calculated average scores for pass and fail groups
- Found the top-performing student

---

### 🔹 Task 2: Matplotlib Visualizations
Generated and saved:
- Bar Chart → Average score per subject  
- Histogram → Distribution of math scores  
- Scatter Plot → Study hours vs average score  
- Box Plot → Attendance comparison (Pass vs Fail)  
- Line Plot → Math vs Science scores  

---

### 🔹 Task 3: Seaborn Visualizations
- Bar plots for Math and Science vs Pass/Fail  
- Scatter plot with regression lines (Attendance vs Average Score)  

**Observation:**
Seaborn is easier for quick and clean visuals, while Matplotlib gives more control.

---

### 🔹 Task 4: Machine Learning
- Features: subject scores, attendance, study hours  
- Target: pass/fail  
- Train-test split (80/20)  
- Feature scaling using StandardScaler  
- Model: Logistic Regression  

Outputs:
- Training accuracy  
- Test accuracy  
- Predictions with correctness (✅ / ❌)  

---

### 🔹 Feature Importance
- Extracted coefficients from model  
- Sorted by importance  
- Visualized using horizontal bar chart  

---

### 🔹 Bonus: New Student Prediction
- Predicted pass/fail for a new student  
- Displayed probability


---

## ▶️ How to Run

1. Install required libraries:
2. Keep `students.csv` in the same folder

3. Run:


---

## ✅ Key Learnings
- Data analysis using Pandas  
- Creating visualizations with Matplotlib and Seaborn  
- Basic machine learning workflow  
- Feature importance interpretation  

---

## 🚀 Conclusion
This project demonstrates a simple end-to-end data analysis and machine learning pipeline. It shows how different factors like study hours and attendance affect student performance.

---

## 📁 Project Files
