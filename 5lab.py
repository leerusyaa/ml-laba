5# lab5.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Set pandas display options
pd.set_option('display.max_columns', 500)


def preproc(df_input):
    # удаляем тяжелые признаки
    drop_cols = ['EDUCATION', 'FACT_ADDRESS_PROVINCE', 'FAMILY_INCOME', 'GEN_INDUSTRY',
                 'GEN_TITLE', 'JOB_DIR', 'MARITAL_STATUS', 'ORG_TP_FCAPITAL', 'REGION_NM',
                 'REG_ADDRESS_PROVINCE', 'ORG_TP_STATE', 'POSTAL_ADDRESS_PROVINCE', 'TP_PROVINCE',
                 'AGREEMENT_RK']
    # Make a copy of data
    df_temp = df_input.copy()
    # Drop the hard columns
    df_temp = df_temp.drop(drop_cols, axis=1, errors='ignore')

    digit_cols = ['LOAN_AVG_DLQ_AMT', 'LOAN_MAX_DLQ_AMT', 'CREDIT', 'FST_PAYMENT', 'PERSONAL_INCOME']
    # Преобр запись в числ признках
    df_temp[digit_cols] = df_temp[digit_cols].replace(regex={',': '.'}).astype('float64')

    return df_temp


def quality_metrics_report(y_true, y_pred):  # истинные и предсказанные ответы
    tp = np.sum((y_true == 1) * (y_pred == 1))
    fp = np.sum((y_true == 0) * (y_pred == 1))
    fn = np.sum((y_true == 1) * (y_pred == 0))
    tn = np.sum((y_true == 0) * (y_pred == 0))

    accuracy = accuracy_score(y_true, y_pred) # (TP + TN) / (TP + FP + FN + TN) общ доля правильных прогнозов
    error_rate = 1 - accuracy
    precision = precision_score(y_true, y_pred)  # tp/(tp+fp) # точность положительных прогнозов
    recall = recall_score(y_true, y_pred)  # tp/(tp+fn) полнота
    f1 = f1_score(y_true, y_pred)  # harmon mean F-мера между precision и recall

    return [tp, fp, fn, tn, accuracy, error_rate, precision, recall, f1]


def main():
    # Load data
    data = pd.read_csv('data_set.csv', delimiter=';')

    # Display basic info about the data
    print("Data shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nData info:")
    print(data.info())

    # Preprocess data
    data_preproc = preproc(data)
    print("\nPreprocessed data info:")
    print(data_preproc.info())

    # Prepare features and target
    label_col = data_preproc.columns == 'TARGET' # [false false false true(target) false ...]
    X = data_preproc.loc[:, ~label_col].values # ~ - побит инверсия, НЕ [false false false true(target) false ...], .loc - все строки, все столбцы кроме TARGET, короч тут велью всех признаков кроме таргета
    y = data_preproc.loc[:, label_col].values.flatten() # ономер массив

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"First 2 rows of X:\n{X[:2]}")
    print(f"First 10 values of y: {y[:10]}")

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Impute missing values
    imp = SimpleImputer(strategy='mean')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    # Scale features
    ss = StandardScaler()
    ss.fit(X_train) # созд объяект для стандартизации и вычисл мю мреднее и сигма стандартн откл кля кд признака
    X_train = ss.transform(X_train) #X_scaled = (X - μ) / σ
    X_test = ss.transform(X_test)
    # все признаки как равнозначные по масштабу

    # Train models
    # K-Nearest Neighbors 
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Decision Tree
    dt = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, #Gini: "Насколько перемешаны классы?, бест выбирает лучшее разделение рез всегда одинаковый
                                min_samples_split=2, min_samples_leaf=10, class_weight=None)
    dt.fit(X_train, y_train)

    # Logistic Regression
    logreg = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight=None) # сред регулязация
    logreg.fit(X_train, y_train)

    # Make predictions
    y_test_knn = knn.predict(X_test)
    y_test_dt = dt.predict(X_test)
    y_test_logreg = logreg.predict(X_test)

    # Display predictions vs truth
    print("\nPredictions vs Truth (first 10):")
    print("Truth  : ", y_test[:10])
    print("kNN    : ", y_test_knn[:10])
    print("DT     : ", y_test_dt[:10])
    print("LogReg : ", y_test_logreg[:10])

    # Get prediction probabilities
    y_test_proba_knn = knn.predict_proba(X_test)[:, 1]
    y_test_proba_dt = dt.predict_proba(X_test)[:, 1]
    y_test_proba_logreg = logreg.predict_proba(X_test)[:, 1]

    print("\nPrediction probabilities (first 10):")
    print("Truth  : ", y_test[:10])
    print("kNN    : ", y_test_proba_knn[:10])
    print("DT     : ", y_test_proba_dt[:10])
    print("LogReg : ", y_test_proba_logreg[:10])

    # Calculate quality metrics
    metrics_report = pd.DataFrame(
        columns=['TP', 'FP', 'FN', 'TN', 'Accuracy', 'Error rate', 'Precision', 'Recall', 'F1'])

    metrics_report.loc['kNN', :] = quality_metrics_report(y_test, y_test_knn)
    metrics_report.loc['DT', :] = quality_metrics_report(y_test, y_test_dt)
    metrics_report.loc['LogReg', :] = quality_metrics_report(y_test, y_test_logreg)

    print("\nQuality Metrics Report:")
    print(metrics_report)

    # Calculate ROC curves and AUC
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_test_proba_knn)
    auc_knn = auc(fpr_knn, tpr_knn)

    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_test_proba_dt)
    auc_dt = auc(fpr_dt, tpr_dt)

    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_test_proba_logreg)
    auc_logreg = auc(fpr_logreg, tpr_logreg)

    # Plot ROC curves
    plt.figure(figsize=(9, 6))
    plt.plot(fpr_knn, tpr_knn, linewidth=3, label='kNN')
    plt.plot(fpr_dt, tpr_dt, linewidth=3, label='DT')
    plt.plot(fpr_logreg, tpr_logreg, linewidth=3, label='LogReg')

    plt.xlabel('FPR', size=18)
    plt.ylabel('TPR', size=18)

    plt.legend(loc='best', fontsize=14)
    plt.grid()
    plt.show()

    print('\nROC AUC Scores:')
    print('kNN ROC AUC    :', auc_knn)
    print('DT ROC AUC     :', auc_dt)
    print('LogReg ROC AUC :', auc_logreg)


if __name__ == "__main__":
    main()