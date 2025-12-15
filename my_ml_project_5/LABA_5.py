import numpy as np
import random 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

X = np.random.randint(0, 2, size=(100, 12))  # 100 примеров, 12 бинарных признаков
Y = np.array([[1, 0] if x.sum() > 6 else [0, 1] for x in X])

# 1. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# 2. Сохранение данных в файлы
np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')

# 3. Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Создание модели MLP с одним скрытым слоем и logsig
model = keras.Sequential([
    keras.layers.Dense(12, activation='sigmoid', input_shape=(12,)),  # скрытый слой
    keras.layers.Dense(2, activation='softmax')  # выходной слой
])

# 5. Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Обучение модели
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# 7. Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Точность на тестовой выборке: {test_acc:.4f}')

# 8. Предсказания
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1) 
y_true_classes = y_test.argmax(axis=1) 

# 9. Метрики
print(classification_report(y_true_classes, y_pred_classes, target_names=['Правящая', 'Оппозиция']))

# 10. Визуализация результатов
# График потерь
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Ошибка на обучении')
plt.plot(history.history['val_loss'], label='Ошибка на валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.title('График потерь')

# Матрица ошибок
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матрица ошибок')
plt.colorbar()
plt.xticks([0, 1], ['Правящая', 'Оппозиция'])
plt.yticks([0, 1], ['Правящая', 'Оппозиция'])
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.show()