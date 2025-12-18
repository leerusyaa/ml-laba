import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Импорт данных
def load_data():
    # Загрузим набор данных по недвижимости в Бостоне
    data = pd.read_csv('boston_house_prices.csv')
    return data

# Функции для работы с деревьями решений
def _get_feature_values(R, feature, feature_names, target_col='target'):
    """Возвращает значения признака и цели для DataFrame или np.ndarray."""
    if isinstance(R, pd.DataFrame):
        return R[feature].values, R[target_col].values
    
    idx = feature_names.index(feature)
    return R[:, idx], R[:, -1]

# Дисперсия
def H(R: np.array, feature_names=None, target_col='target') -> float:
    """Вычислить критерий информативности для фиксированного набора объектов R."""
    if isinstance(R, pd.DataFrame):
        y_vals = R[target_col].values
    else:
        y_vals = R[:, -1]
    
    if len(y_vals) == 0:
        return 0.0
    
    y_mean = np.mean(y_vals)
    return np.mean((y_vals - y_mean) ** 2)

# Разделение вершины
def split_node(R: np.array, feature: str, t: float, feature_names: List[str], target_col='target') -> Iterable[np.array]:
    """Разделить фиксированный набор объектов R по признаку feature с пороговым значением t"""
    if isinstance(R, pd.DataFrame):
        R_left = R[R[feature] <= t].copy()
        R_right = R[R[feature] > t].copy()
        return R_left, R_right
    
    idx = feature_names.index(feature)
    mask = R[:, idx] <= t
    return R[mask], R[~mask]

# Функционал качества
def Q(R: np.array, feature: str, t: float, feature_names: List[str], target_col='target') -> float:
    """Вычислить функционал качества для заданных параметров разделения"""
    R_left, R_right = split_node(R, feature, t, feature_names, target_col)
    
    n = len(R)
    n_left = len(R_left)
    n_right = len(R_right)
    
    return (n_left / n) * H(R_left, feature_names, target_col) + (n_right / n) * H(R_right, feature_names, target_col)

# Получение кандидатов порогов
def get_candidate_thresholds(R, feature: str, feature_names: List[str], target_col='target') -> List[float]:
    """Пороги — середины между соседними уникальными значениями признака."""
    feat_vals, _ = _get_feature_values(R, feature, feature_names, target_col)
    uniq = np.unique(feat_vals)
    
    if len(uniq) == 1:
        return [uniq[0]]
    
    return ((uniq[:-1] + uniq[1:]) / 2).tolist()

# Поиск оптимального разбиения
def get_optimal_split(R: np.array, feature: str, feature_names: List[str], target_col='target') -> Tuple[float, List[float], List[float]]:
    """Найти оптимальное разбиение для данного признака"""
    thresholds = get_candidate_thresholds(R, feature, feature_names, target_col)
    q_values = [Q(R, feature, t, feature_names, target_col) for t in thresholds]
    
    best_index = int(np.argmin(q_values))
    best_t = thresholds[best_index]
    
    return best_t, thresholds, q_values

# Визуализация зависимости Q(t)
def plot_q_vs_threshold(feature: str, X_train: pd.DataFrame, feature_names: List[str], target_col='target'):
    """Построить график критерия ошибки в зависимости от значения порога"""
    best_t, thresholds, q_vals = get_optimal_split(X_train, feature, feature_names, target_col)
    
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, q_vals)
    plt.xlabel("threshold t")
    plt.ylabel(f"Q(R, {feature}, t)")
    plt.title(f"Зависимость Q(t) для признака {feature}")
    plt.grid(True)
    plt.show()
    
    print(f"Лучший порог для {feature}:", best_t)
    print("Минимальное значение Q:", min(q_vals))
    
    return best_t, thresholds, q_vals

# Поиск лучшего признака для первого разбиения
def find_best_feature(X_train: pd.DataFrame, feature_names: List[str], target_col='target'):
    """Найти признак, показывающий наилучшее качество"""
    best_feature = None
    best_feature_t = None
    best_feature_Q = np.inf
    
    for feat in feature_names:
        t_opt, thr, qv = get_optimal_split(X_train, feat, feature_names, target_col)
        q_min = min(qv)
        
        if q_min < best_feature_Q:
            best_feature_Q = q_min
            best_feature = feat
            best_feature_t = t_opt
            thresholds_best = thr
            q_vals_best = qv
    
    print("Лучший признак:", best_feature)
    print("Лучший порог:", best_feature_t)
    print("Лучшее значение Q:", best_feature_Q)
    
    return best_feature, best_feature_t, thresholds_best, q_vals_best

# Визуализация разбиения
def visualize_split(X_train: pd.DataFrame, feature: str, threshold: float, target_col='target'):
    """Визуализировать разбиение на диаграмме рассеяния"""
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train[feature], X_train[target_col])
    plt.axvline(threshold, color='red', linestyle='--')
    plt.xlabel(feature)
    plt.ylabel(f"{target_col} (target)")
    plt.title(f"Разбиение по лучшему признаку")
    plt.grid(True)
    plt.show()

# Обучение модели с помощью DecisionTreeRegressor
def train_decision_tree(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: List[str], target_col='target'):
    """Обучить модель дерева решений и вывести результаты"""
    X_train_feat = X_train.drop(columns=[target_col])
    X_test_feat = X_test.drop(columns=[target_col])
    
    y_train = X_train[target_col]
    y_test = X_test[target_col]
    
    tree = DecisionTreeRegressor(random_state=13)
    tree.fit(X_train_feat, y_train)
    
    print("R² на train:", tree.score(X_train_feat, y_train))
    print("R² на test:", tree.score(X_test_feat, y_test))
    
    print("\nВажности признаков:")
    for f, imp in zip(feature_names, tree.feature_importances_):
        print(f, ":", imp)
    
    return tree, X_train_feat, X_test_feat, y_train, y_test

# Основная функция
def main():
    # Загрузка данных
    data = load_data()
    
    # Определение признаков
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    target_col = 'MEDV'
    
    # Создание DataFrame
    X = pd.DataFrame(data, columns=feature_names, index=range(len(data)))
    y = pd.DataFrame(data, columns=[target_col], index=range(len(data)))
    
    # Добавление целевой переменной в X для совместимости
    X[target_col] = y
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=13)
    
    print("=" * 50)
    print("1. Подсчет критерия ошибки и функции разбиения")
    print("=" * 50)
    
    # Демонстрация работы функций H, split_node и Q
    print("\nФункции H, split_node и Q реализованы и готовы к использованию")
    
    print("\n" + "=" * 50)
    print("2. Поиск оптимального разбиения для признака RM")
    print("=" * 50)
    
    # Построение графика Q(t) для признака RM
    best_t_rm, thresholds_rm, q_vals_rm = plot_q_vs_threshold('RM', X_train, feature_names, target_col)
    
    print("\n" + "=" * 50)
    print("3. Поиск лучшего признака для первого разбиения")
    print("=" * 50)
    
    # Поиск лучшего признака
    best_feature, best_feature_t, thresholds_best, q_vals_best = find_best_feature(X_train, feature_names, target_col)
    
    print("\n" + "=" * 50)
    print("4. Визуализация разбиения")
    print("=" * 50)
    
    # Визуализация разбиения
    visualize_split(X_train, best_feature, best_feature_t, target_col)
    
    print("\n" + "=" * 50)
    print("5. Обучение модели DecisionTreeRegressor")
    print("=" * 50)
    
    # Обучение модели дерева решений
    tree, X_train_feat, X_test_feat, y_train, y_test = train_decision_tree(X_train, X_test, feature_names, target_col)
    
    print("\n" + "=" * 50)
    print("Выполнение завершено!")
    print("=" * 50)

if __name__ == "__main__":
    main()