import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score

# Загрузка данных
df = pd.read_csv('D:/учеба/3 курс/ии/git1/my_ml_project/processed_world_economics.csv')

median_gdp = df['GDP'].median()
df['gdp_class'] = (df['GDP'] > median_gdp).astype(int)

target_column = 'gdp_class'

# Экономически значимые признаки для предсказания ВВП
economic_features = [
    'Inflation Rate',    
    'Jobless Rate',           
    'Current Account',   
    'Gov. Budget',       
    'Debt/GDP',          
    'Population'         
]

X = df[economic_features] 
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# СЛУЧАЙНЫЙ ЛЕС С OOB ОЦЕНКОЙ
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, oob_score=True, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest:")
print(f"OOB Score (Accuracy): {rf_model.oob_score_:.4f}")
print(f"OOB Error: {1 - rf_model.oob_score_:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# ADABOOST
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
y_pred_proba_ada = ada_model.predict_proba(X_test)[:, 1]

print("\nADABOOST:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ada):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ada))

# ГРАДИЕНТНЫЙ БУСТИНГ
gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]

print("\nGradient Boosting:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))

# ROC-КРИВЫЕ
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_ada, tpr_ada, _ = roc_curve(y_test, y_pred_proba_ada)
roc_auc_ada = auc(fpr_ada, tpr_ada)

fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

print("AUC Scores:")
print(f"Random Forest: {roc_auc_rf:.4f}")
print(f"AdaBoost: {roc_auc_ada:.4f}")
print(f"Gradient Boosting: {roc_auc_gb:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, marker='o', label=f'Random Forest')
plt.plot(fpr_ada, tpr_ada, marker='o', label=f'AdaBoost')
plt.plot(fpr_gb, tpr_gb, marker='o', label=f'Gradient Boosting')
plt.ylim([0,1.1]) 
plt.xlim([0,1.1]) 
plt.ylabel('TPR') 
plt.xlabel('FPR') 
plt.title(f'ROC curve')
plt.legend(loc="lower right")
plt.show()

# МАТРИЦЫ ОШИБОК
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
           xticklabels=['Низкий ВВП', 'Высокий ВВП'], 
           yticklabels=['Низкий ВВП', 'Высокий ВВП'])
axes[0].set_title('Confusion Matrix - Random Forest')  
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# AdaBoost
cm_ada = confusion_matrix(y_test, y_pred_ada)
sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Blues', ax=axes[1],
           xticklabels=['Низкий ВВП', 'Высокий ВВП'], 
           yticklabels=['Низкий ВВП', 'Высокий ВВП'])
axes[1].set_title('Confusion Matrix - AdaBoost')  
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

# Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', ax=axes[2],
           xticklabels=['Низкий ВВП', 'Высокий ВВП'], 
           yticklabels=['Низкий ВВП', 'Высокий ВВП'])
axes[2].set_title('Confusion Matrix - Gradient Boosting')  
axes[2].set_xlabel('Predicted Label')
axes[2].set_ylabel('True Label')
plt.tight_layout()
plt.show()