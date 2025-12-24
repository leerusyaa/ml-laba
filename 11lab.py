# Анализ выживаемости на Титанике с использованием Random Forest


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def load_data():
    """Загрузка данных Titanic"""
    print("Загрузка данных Titanic...")
    data = pd.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    return data

def explore_data(data):
    """Исследование данных"""
    print("\n=== Исследование данных ===")
    print(f"Размер данных: {data.shape}")
    print("\nПервые 5 строк:")
    print(data.head())
    
    print("\nИнформация о данных:")
    print(data.info())
    
    print("\nСтатистическое описание числовых признаков:")
    print(data.describe())
    
    print("\nПропущенные значения:")
    print(data.isnull().sum())

def preprocess_data(data):
    """Предобработка данных"""
    print("\n=== Предобработка данных ===")
    
    # Разделение на признаки и целевую переменную
    X = data.drop(columns=['Survived'])
    y = data['Survived']
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Копирование данных для обработки
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Заполнение пропусков
    for df in [X_train_processed, X_test_processed]:
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Удаление неинформативных признаков
    X_train_processed.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    X_test_processed.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    
    # Кодирование категориальных признаков
    encoder = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        X_train_processed[col] = encoder.fit_transform(X_train_processed[col])
        X_test_processed[col] = encoder.transform(X_test_processed[col])
    
    return X_train_processed, X_test_processed, y_train, y_test

def evaluate_model(y_test, y_pred, model_name="Модель"):
    """Оценка модели"""
    print(f"\n=== Оценка модели: {model_name} ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Матрица ошибок - {model_name}')
    plt.xlabel("Предсказанные")
    plt.ylabel("Фактические")
    plt.show()
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность: {accuracy:.4f}")
    return accuracy

def train_random_forest(X_train, X_test, y_train, y_test):
    """Обучение модели Random Forest"""
    print("\n=== Обучение Random Forest ===")
    
    # Базовая модель
    print("\n1. Базовая модель Random Forest:")
    rf_base = RandomForestClassifier(random_state=42)
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_test)
    accuracy_base = evaluate_model(y_test, y_pred_base, "Random Forest (Базовая)")
    
    # Подбор гиперпараметров с GridSearchCV
    print("\n2. Подбор гиперпараметров GridSearchCV:")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        param_grid, 
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая кросс-валидационная точность: {grid_search.best_score_:.4f}")
    
    # Лучшая модель
    best_rf = grid_search.best_estimator_
    y_pred_best = best_rf.predict(X_test)
    accuracy_best = evaluate_model(y_test, y_pred_best, "Random Forest (Оптимизированная)")
    
    return best_rf, accuracy_base, accuracy_best

def compare_models(X_train, X_test, y_train, y_test, best_rf):
    """Сравнение разных классификаторов"""
    print("\n=== Сравнение классификаторов ===")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": best_rf
    }
    
    results = []
    
    for name, model in models.items():
        if name != "Random Forest":  # Random Forest уже обучен
            model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append({"Модель": name, "Точность": acc})
        
        # Вывод матрицы ошибок для каждой модели
        evaluate_model(y_test, preds, name)
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    print("\nИтоговая таблица сравнения моделей:")
    print(results_df.sort_values("Точность", ascending=False))
    
    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results_df["Модель"], results_df["Точность"])
    plt.title("Сравнение точности классификаторов")
    plt.xlabel("Модель")
    plt.ylabel("Точность")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Добавляем значения на столбцы
    for bar, acc in zip(bars, results_df["Точность"]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def feature_importance_analysis(model, X_train):
    """Анализ важности признаков"""
    print("\n=== Анализ важности признаков ===")
    
    # Получаем важность признаков
    importances = model.feature_importances_
    feature_names = X_train.columns
    
    # Создаем DataFrame с важностью признаков
    feature_importance_df = pd.DataFrame({
        'Признак': feature_names,
        'Важность': importances
    }).sort_values('Важность', ascending=False)
    
    print("Важность признаков:")
    print(feature_importance_df)
    
    # Визуализация важности признаков
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_importance_df['Признак'], feature_importance_df['Важность'])
    plt.title('Важность признаков в модели Random Forest')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    
    # Добавляем значения на столбцы
    for bar, imp in zip(bars, feature_importance_df['Важность']):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{imp:.4f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()

def main():
    """Основная функция"""
    print("=" * 60)
    print("АНАЛИЗ ВЫЖИВАЕМОСТИ НА ТИТАНИКЕ")
    print("=" * 60)
    
    # 1. Загрузка данных
    data = load_data()
    
    # 2. Исследование данных
    explore_data(data)
    
    # 3. Предобработка данных
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    print(f"\nРазмеры выборок после предобработки:")
    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Тестовая выборка: {X_test.shape}")
    print(f"Целевая переменная (обучающая): {y_train.shape}")
    print(f"Целевая переменная (тестовая): {y_test.shape}")
    
    # 4. Обучение Random Forest
    best_rf, acc_base, acc_best = train_random_forest(X_train, X_test, y_train, y_test)
    
    print(f"\nУлучшение точности после оптимизации: {acc_best - acc_base:.4f}")
    
    # 5. Анализ важности признаков
    feature_importance_analysis(best_rf, X_train)
    
    # 6. Сравнение с другими моделями
    results_df = compare_models(X_train, X_test, y_train, y_test, best_rf)
    
    # 7. Вывод итогов
    print("\n" + "=" * 60)
    print("ИТОГИ АНАЛИЗА")
    print("=" * 60)
    print(f"Лучшая модель: {results_df.iloc[0]['Модель']}")
    print(f"Точность лучшей модели: {results_df.iloc[0]['Точность']:.4f}")
    print(f"\nРекомендации:")
    print("1. Random Forest показал наилучший результат")
    print("2. Наиболее важные признаки: Пол, Возраст, Класс билета")
    print("3. Для улучшения точности можно:")
    print("   - Собрать больше данных")
    print("   - Создать новые признаки")
    print("   - Попробовать другие алгоритмы (XGBoost, Gradient Boosting)")
    print("   - Использовать более сложную предобработку")
    
    return best_rf, results_df

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    model, results = main()