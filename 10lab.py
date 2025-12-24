import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, r2_score
import seaborn as sns

# Для градиентного бустинга
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Для ансамблевых методов
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, StackingClassifier

# ===========================
# Загрузка и подготовка данных
# ===========================
print("Загрузка датасета Olivetti Faces...")
faces = fetch_olivetti_faces()
images = faces.images  # Изображения размером 64x64
labels = faces.target  # Метки классов (номера людей)

print(f"Размер датасета: {images.shape}")
print(f"Количество классов: {len(np.unique(labels))}")
print(f"Размер изображения: {images.shape[1]}x{images.shape[2]}")

# Преобразование изображений в векторы
X = images.reshape(images.shape[0], -1)  # Преобразуем в (n_samples, 4096)
y = labels

# Нормализация значений пикселей
X = X / 255.0

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")
print()

# ===========================
# ЗАДАНИЕ 1: Градиентный бустинг
# ===========================
print("=" * 60)
print("ЗАДАНИЕ 1: Классификация лиц с помощью градиентного бустинга")
print("=" * 60)

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Обучение и оценка модели"""
    start_time = time.time()
    
    # Обучение модели
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Предсказание
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{model_name}:")
    print(f"  Точность (accuracy): {accuracy:.4f}")
    print(f"  Время обучения: {train_time:.2f} сек")
    print(f"  Время предсказания: {predict_time:.2f} сек")
    print()
    
    return accuracy, train_time, predict_time, y_pred

# Создание моделей градиентного бустинга
models = {
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    ),
    "GradientBoosting (sklearn)": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

# Обучение и оценка моделей
results = {}
for name, model in models.items():
    accuracy, train_time, predict_time, y_pred = train_and_evaluate_model(
        model, name, X_train, X_test, y_train, y_test
    )
    results[name] = {
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
        'y_pred': y_pred
    }

# Сравнение моделей
print("\nСравнение моделей градиентного бустинга:")
print("-" * 50)
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"Лучшая модель: {best_model[0]} с точностью {best_model[1]['accuracy']:.4f}")

# Вывод отчета классификации для лучшей модели
print(f"\nОтчет классификации для {best_model[0]}:")
print(classification_report(y_test, best_model[1]['y_pred'], zero_division=0))

# Матрица ошибок для лучшей модели
fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(y_test, best_model[1]['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap='Blues')
plt.title(f'Матрица ошибок: {best_model[0]}')
plt.tight_layout()
plt.show()

# ===========================
# ЗАДАНИЕ 2: VotingClassifier
# ===========================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 2: Классификация лиц с помощью VotingClassifier")
print("=" * 60)

# Создание базовых классификаторов
base_models = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB())
]

# Обучение и оценка отдельных моделей
print("Оценка отдельных базовых моделей:")
individual_results = {}
for name, model in base_models:
    accuracy, train_time, predict_time, y_pred = train_and_evaluate_model(
        model, name, X_train, X_test, y_train, y_test
    )
    individual_results[name] = accuracy

# Создание VotingClassifier (мягкое голосование)
voting_clf = VotingClassifier(
    estimators=base_models,
    voting='soft'  # Используем мягкое голосование
)

# Обучение и оценка VotingClassifier
accuracy_voting, train_time_voting, predict_time_voting, y_pred_voting = train_and_evaluate_model(
    voting_clf, "VotingClassifier (soft)", X_train, X_test, y_train, y_test
)

# Сравнение VotingClassifier с отдельными моделями
print("\nСравнение точности:")
print("-" * 30)
for name, accuracy in individual_results.items():
    print(f"{name}: {accuracy:.4f}")
print(f"VotingClassifier: {accuracy_voting:.4f}")

if accuracy_voting > max(individual_results.values()):
    print("\nVotingClassifier показал лучший результат!")
else:
    print("\nЛучший результат показала отдельная модель.")

# Матрица ошибок для VotingClassifier
fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred_voting)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap='Greens')
plt.title('Матрица ошибок: VotingClassifier')
plt.tight_layout()
plt.show()

# ===========================
# ЗАДАНИЕ 3: StackingClassifier
# ===========================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 3: Классификация лиц с помощью StackingClassifier")
print("=" * 60)

# Создание StackingClassifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5  # Используем кросс-валидацию для обучения метаклассификатора
)

# Обучение и оценка StackingClassifier
accuracy_stacking, train_time_stacking, predict_time_stacking, y_pred_stacking = train_and_evaluate_model(
    stacking_clf, "StackingClassifier", X_train, X_test, y_train, y_test
)

# Сравнение всех ансамблевых методов
print("\nСравнение ансамблевых методов:")
print("-" * 40)
print(f"Лучшая отдельная модель: {max(individual_results.items(), key=lambda x: x[1])}")
print(f"VotingClassifier: {accuracy_voting:.4f}")
print(f"StackingClassifier: {accuracy_stacking:.4f}")

if accuracy_stacking > accuracy_voting:
    print("\nStackingClassifier показал лучший результат среди ансамблевых методов!")
elif accuracy_voting > accuracy_stacking:
    print("\nVotingClassifier показал лучший результат среди ансамблевых методов!")
else:
    print("\nОба ансамблевых метода показали одинаковый результат.")

# Матрица ошибок для StackingClassifier
fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred_stacking)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap='Oranges')
plt.title('Матрица ошибок: StackingClassifier')
plt.tight_layout()
plt.show()

# ===========================
# ЗАДАНИЕ 4: Предсказание нижней половины лица
# ===========================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 4: Предсказание нижней половины лица по верхней")
print("=" * 60)

# Подготовка данных для регрессии
# Разделение изображений на верхнюю и нижнюю половины
upper_half = images[:, :32, :]  # Верхняя половина (первые 32 строки)
lower_half = images[:, 32:, :]  # Нижняя половина (последние 32 строки)

# Преобразование 2D изображений в 1D векторы
X_reg = upper_half.reshape((images.shape[0], -1))
y_reg = lower_half.reshape((images.shape[0], -1))

# Нормализация
X_reg = X_reg / 255.0
y_reg = y_reg / 255.0

# Разделение на обучающую и тестовую выборки
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"Размерность признаков: {X_train_reg.shape[1]}")
print(f"Размерность целевой переменной: {y_train_reg.shape[1]}")

# Создание регрессионных моделей
reg_models = {
    "LightGBM Regressor": lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1
    ),
    "XGBoost Regressor": xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    "GradientBoosting Regressor": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

# Функция для обучения и оценки регрессионных моделей
def train_and_evaluate_regressor(model, model_name, X_train, X_test, y_train, y_test):
    """Обучение и оценка регрессионной модели"""
    start_time = time.time()
    
    # Обучение модели
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Предсказание
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # Оценка качества (R²)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name}:")
    print(f"  R² score: {r2:.4f}")
    print(f"  Время обучения: {train_time:.2f} сек")
    print(f"  Время предсказания: {predict_time:.2f} сек")
    print()
    
    return r2, train_time, predict_time, y_pred

# Обучение и оценка регрессионных моделей
reg_results = {}
for name, model in reg_models.items():
    r2, train_time, predict_time, y_pred = train_and_evaluate_regressor(
        model, name, X_train_reg, X_test_reg, y_train_reg, y_test_reg
    )
    reg_results[name] = {
        'r2': r2,
        'train_time': train_time,
        'predict_time': predict_time,
        'y_pred': y_pred
    }

# Сравнение регрессионных моделей
print("\nСравнение регрессионных моделей:")
print("-" * 40)
best_reg_model = max(reg_results.items(), key=lambda x: x[1]['r2'])
print(f"Лучшая модель: {best_reg_model[0]} с R² = {best_reg_model[1]['r2']:.4f}")

# Визуализация результатов регрессии
def visualize_predictions(original_lower, predicted_lower, model_name, n_samples=3):
    """Визуализация оригинальной и предсказанной нижней половины лица"""
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3 * n_samples))
    
    for i in range(min(n_samples, len(original_lower))):
        # Оригинальная нижняя половина
        axes[i, 0].imshow(original_lower[i].reshape(32, 64), cmap='gray')
        axes[i, 0].set_title(f'Оригинал (образец {i+1})')
        axes[i, 0].axis('off')
        
        # Предсказанная нижняя половина
        axes[i, 1].imshow(predicted_lower[i].reshape(32, 64), cmap='gray')
        axes[i, 1].set_title(f'Предсказание {model_name}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Визуализация для лучшей регрессионной модели
print(f"\nВизуализация результатов для {best_reg_model[0]}:")
visualize_predictions(
    y_test_reg,
    best_reg_model[1]['y_pred'],
    best_reg_model[0],
    n_samples=3
)

# ===========================
# ОБЩИЕ ВЫВОДЫ
# ===========================
print("\n" + "=" * 60)
print("ОБЩИЕ ВЫВОДЫ")
print("=" * 60)

print("\n1. Сравнение алгоритмов градиентного бустинга:")
print("   - LightGBM обычно быстрее обучается")
print("   - XGBoost часто показывает хороший баланс точности и времени")
print("   - GradientBoosting из sklearn проще в использовании")

print("\n2. Эффективность ансамблевых методов:")
print("   - VotingClassifier: улучшает стабильность предсказаний")
print("   - StackingClassifier: может достигать более высокой точности за счет мета-обучения")
print("   - Ансамбли особенно эффективны, когда базовые модели разнообразны")

print("\n3. Задача регрессии (предсказание лица):")
print("   - R² score показывает, насколько хорошо модели восстанавливают изображение")
print("   - Задача сложная из-за высокой размерности целевой переменной")
print("   - Визуализация помогает оценить качество восстановления")

print("\n4. Рекомендации для данного датасета:")
print("   - Для классификации: StackingClassifier или XGBoost")
print("   - Для быстрого прототипирования: VotingClassifier")
print("   - Для задач восстановления: LightGBM Regressor (хороший баланс скорости и качества)")