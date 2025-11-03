import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV 
from sklearn.metrics import (mean_squared_error, accuracy_score, 
confusion_matrix, classification_report, root_mean_squared_error, 
mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler

dff = pd.read_csv('D:/учеба/3 курс/ии/git1/my_ml_project/processed_world_economics.csv') 

# Оставляем только числовые поля
df = dff.select_dtypes(include=['number'])   

# 1. Задача регрессии 
X = df.drop(['GDP'], axis=1)    
y = df['GDP'] 

print(list(X)) 

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Линейная регрессия
linear_model = LinearRegression() 
linear_model.fit(X_train, y_train)   
y_pred_lin = linear_model.predict(X_test) 

print("Линейная регрессия:")  
print(f'R^2: {r2_score(y_test, y_pred_lin):.2f}')   
print(f'MAE: {mean_absolute_error(y_test, y_pred_lin):.2f}')  
print(f'RMSE: {root_mean_squared_error(y_test, y_pred_lin):.2f}')
print(f'MSE: {mean_squared_error(y_test, y_pred_lin):.2f}') 

# 2. Задача классификации 
median_train = y_train.median()       
y_train_class = (y_train > median_train).astype(int)   
y_test_class = (y_test > median_train).astype(int)    

# Логистическая регрессия
logreg_model = LogisticRegression() 
logreg_model.fit(X_train, y_train_class)  
y_pred_log = logreg_model.predict(X_test)  

print("\nЛогистическая регрессия:") 
print(f'Accuracy: {accuracy_score(y_test_class, y_pred_log):.2f}')   
print(f'Classification Report:\n{classification_report(y_test_class, y_pred_log)}')  

# Матрица ошибок для классификации 
cm = confusion_matrix(y_test_class, y_pred_log)  
plt.figure(figsize=(5, 4)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Низкий ВВП', 'Высокий ВВП'], yticklabels=['Низкий ВВП', 'Высокий ВВП']) 
plt.title('Confusion Matrix - Логистическая регрессия')  
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
