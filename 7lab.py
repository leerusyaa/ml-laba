import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    print("Лабораторная работа: Сравнение KNN и логистической регрессии на данных Iris")
    print("=" * 70)

    # Загрузка данных Iris
    print("\n1. Загрузка данных Iris...")
    iris = load_iris(as_frame=True)
    data = iris['data']
    y = iris['target'].values

    print(f"Размер данных: {data.shape}")
    print(f"Количество классов: {len(np.unique(y))}")

    # Задание 1: Ищем versicolor (класс 1)
    print("\n2. Преобразование задачи в бинарную классификацию (versicolor vs остальные)...")
    y = (y == 1).astype(int)  # 1 для versicolor, 0 для остальных
    print(f"Количество цветков versicolor: {np.sum(y == 1)}")
    print(f"Количество цветков других классов: {np.sum(y == 0)}")

    # Задание 2: Подготовка данных
    print("\n3. Подготовка данных для обучения...")
    # Используем только два признака для визуализации
    X = data[['sepal length (cm)', 'sepal width (cm)']]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    # Нормализация данных
    print("\n4. Нормализация данных...")
    scaler = StandardScaler()
    scaler.fit(X_train, y_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Визуализация данных
    print("\n5. Визуализация обучающих данных...")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=y_train, cmap='viridis', edgecolors='k', alpha=0.8)
    plt.xlabel('Длина чашелистика (стандартизированная)')
    plt.ylabel('Ширина чашелистика (стандартизированная)')
    plt.title('Визуализация обучающих данных Iris (versicolor vs остальные)')
    plt.colorbar(label='Класс (0 - не versicolor, 1 - versicolor)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Задание 3: Обучение моделей
    print("\n6. Обучение классификаторов...")

    # K-ближайших соседей
    knn = KNeighborsClassifier(n_neighbors=5)
    # Логистическая регрессия
    logreg = LogisticRegression(random_state=123)

    # Обучение моделей
    knn.fit(X_train_scaled, y_train)
    logreg.fit(X_train_scaled, y_train)

    print("Модели успешно обучены!")

    # Получение прогнозов
    print("\n7. Прогнозирование на тестовой выборке...")
    y_pred_knn = knn.predict(X_test_scaled)
    y_pred_logreg = logreg.predict(X_test_scaled)

    # Получение вероятностей
    y_proba_knn = knn.predict_proba(X_test_scaled)
    y_proba_logreg = logreg.predict_proba(X_test_scaled)

    print("\nПервые 5 прогнозов KNN:")
    print(y_pred_knn[:5])

    print("\nПервые 5 вероятностей KNN:")
    print(y_proba_knn[:5])

    print("\nПервые 5 прогнозов логистической регрессии:")
    print(y_pred_logreg[:5])

    print("\nПервые 5 вероятностей логистической регрессии:")
    print(y_proba_logreg[:5])

    # Оценка качества моделей
    print("\n8. Оценка точности классификаторов...")
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

    print(f"Точность KNN: {accuracy_knn:.4f}")
    print(f"Точность логистической регрессии: {accuracy_logreg:.4f}")

    # Визуализация разделяющих поверхностей
    print("\n9. Визуализация разделяющих поверхностей...")

    # Создаем сетку для построения разделяющих поверхностей
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))



    # График для KNN
    ax1 = axes[0]
    plot_decision_regions(X_train_scaled, y_train, clf=knn, legend=2, ax=ax1)
    ax1.set_title('Разделяющая поверхность KNN (K=5)')
    ax1.set_xlabel('Длина чашелистика (стандартизированная)')
    ax1.set_ylabel('Ширина чашелистика (стандартизированная)')
    ax1.grid(True, alpha=0.3)

    # График для логистической регрессии
    ax2 = axes[1]
    plot_decision_regions(X_train_scaled, y_train, clf=logreg, legend=2, ax=ax2)
    ax2.set_title('Разделяющая поверхность логистической регрессии')
    ax2.set_xlabel('Длина чашелистика (стандартизированная)')
    ax2.set_ylabel('Ширина чашелистика (стандартизированная)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Анализ результатов
    print("\n10. Анализ результатов...")
    print(f"\nСравнение точности моделей:")
    print(f"KNN: {accuracy_knn * 100:.2f}%")
    print(f"Логистическая регрессия: {accuracy_logreg * 100:.2f}%")

    if accuracy_knn > accuracy_logreg:
        print("\nKNN показал лучшую точность на данной задаче.")
    elif accuracy_logreg > accuracy_knn:
        print("\nЛогистическая регрессия показала лучшую точность на данной задаче.")
    else:
        print("\nМодели показали одинаковую точность.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()