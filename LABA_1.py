import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

df = pd.read_csv('D:/учеба/3 курс/ии/git1/my_ml_project/world_economics.csv') # загрузка данных 
print(df.info()) #информация о датасете
print(df.dtypes) #типы данных

cols = df.columns # получение названий колонок
print(cols.tolist()) 

print("Количество пропущенных значений по столбцам:")
nan_matrix = df.isnull() 
missing_values_count = nan_matrix.sum() 
print(missing_values_count) 

# заполняем числовые столбцы (используем медиану для числовых данных)
numeric_df = df.select_dtypes(include='number')
for col in numeric_df.columns:
    if df[col].isnull().sum() > 0:
        age_median = df[col].median()
        df[col] = df[col].fillna(age_median) 

# заполняем категориальные столбцы (используем моду для категориальных данных)
categorical_df = df.select_dtypes(exclude='number') 
for col in categorical_df.columns:
    if df[col].isnull().sum() > 0:
        cabin_mode = df[col].mode()[0] 
        df[col] = df[col].fillna(cabin_mode) 

print("\nКоличество пропущенных значений после заполнения:") 
print(df.isnull().sum())  

# нормализация данных
scaler_minmax = MinMaxScaler() 

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns # выбор числовых столбцов для нормализации
df[numeric_columns] = scaler_minmax.fit_transform(df[numeric_columns])

print("\nДанные после нормализации:")
print(df[numeric_columns].head()) 

# преобразование категориальных данных
categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("\nДанные после преобразования категориальных признаков:")
print(df.head()) 
print(df.columns)  

# сохраняем обработанные данные в CSV-файл
df.to_csv("processed_world_economics.csv", index=False)  

