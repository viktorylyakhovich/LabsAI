import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import roc_curve, auc, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

dff = pd.read_csv('D:/учеба/3 курс/ии/git1/my_ml_project/processed_world_economics.csv') 

# Создаем бинарный признак для классификации на основе GDP
median_gdp = dff['GDP'].median()
dff['High_GDP'] = (dff['GDP'] > median_gdp).astype(int) 
 
df = dff.select_dtypes(include=['number'])  

# Определяем целевые переменные
reg_target = 'GDP'
class_target = 'High_GDP'

# ЗАДАЧА РЕГРЕССИИ
# Определяем какие колонки исключить
exclude_cols = [reg_target, class_target]

# Создаем X и y для регрессии
X_reg = df.drop(exclude_cols, axis=1)
y_reg = df[reg_target]

# Разделение на обучающую и тестовую выборки для регрессии
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Создание и обучение модели дерева решений для регрессии
dt_regressor_model = DecisionTreeRegressor(max_depth=4, random_state=42)
dt_regressor_model.fit(X_reg_train, y_reg_train)

y_reg_pred = dt_regressor_model.predict(X_reg_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)

print(f"\nРегрессия:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")

# ЗАДАЧА КЛАССИФИКАЦИИ
X_clf = df.drop(exclude_cols, axis=1)
y_clf = df[class_target]

# Разделение для классификации
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf)

# Модель классификации
dt_classifier_model = DecisionTreeClassifier(max_depth=6, max_leaf_nodes = 5, random_state=42)
dt_classifier_model.fit(X_clf_train, y_clf_train)

y_clf_pred = dt_classifier_model.predict(X_clf_test)
y_proba = dt_classifier_model.predict_proba(X_clf_test)

accuracy = accuracy_score(y_clf_test, y_clf_pred)
print(f"\nКлассификация:")
print(f"Accuracy: {accuracy:.4f}")

# ROC-кривая
fpr, tpr, thresholds = roc_curve(y_clf_test, y_proba[:, 1])
auc_metric = auc(fpr, tpr)
print(f"ROC-AUC: {auc_metric:.4f}")

# Визуализация ROC-кривой 
plt.plot(fpr, tpr, marker='o') 
plt.ylim([0,1.1]) 
plt.xlim([0,1.1]) 
plt.ylabel('TPR') 
plt.xlabel('FPR') 
plt.title(f'ROC curve')
plt.show()

# Визуализация деревьев решений
tree.plot_tree(dt_regressor_model)
plt.title('Дерево решений для регрессии GDP')
plt.show()

# Визуализация дерева классификации
tree.plot_tree(dt_classifier_model)
plt.title('Дерево решений для классификации High GDP')
plt.show()