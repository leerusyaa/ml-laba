import pandas as pd

ratings = pd.read_csv('music_listening.csv', low_memory=False)
print("Оригинальная форма:", ratings.shape)

ratings = ratings.T
print("Форма после транспонирования:", ratings.shape)

ratings = ratings[ratings.index != 'user']
print(f'Количество строк после удаления user: {len(ratings)}')

ratings = ratings[ratings.index != 'the beatles']
print(f'Количество строк после удаления the beatles: {len(ratings)}')

has_coldplay = 'coldplay' in ratings.index
if has_coldplay:
    vec_coldplay_raw = ratings.loc['coldplay'].copy()

has_the_beatles = 'the beatles' in ratings.index
if has_the_beatles:
    vec_the_beatles_raw = ratings.loc['the beatles'].copy()

artists_to_remove = []
if has_coldplay:
    artists_to_remove.append('coldplay')
if has_the_beatles:
    artists_to_remove.append('the beatles')

if artists_to_remove:
    print(f"Удаляем строки для кластеризации: {artists_to_remove}")
    ratings_for_clustering = ratings.drop(index=artists_to_remove)
else:
    ratings_for_clustering = ratings

print(f'Количество строк в ratings для кластеризации: {len(ratings_for_clustering)}')

ratings_for_clustering = ratings_for_clustering.replace(',', '.', regex=True)
ratings_for_clustering = ratings_for_clustering.apply(pd.to_numeric, errors='coerce')

ratings_for_clustering = ratings_for_clustering.fillna(0)

from sklearn.preprocessing import normalize
ratings_normalized = normalize(ratings_for_clustering.values, norm='l2', axis=1)

# KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(ratings_normalized)

# Центроиды
centroids = kmeans.cluster_centers_
print("Центроиды (первых 3 значения для примера):")
for i, c in enumerate(centroids):
    print(f"  Кластер {i}: {c[:3]}...")

if has_the_beatles and has_coldplay:
    vec1_raw = vec_the_beatles_raw
    vec2_raw = vec_coldplay_raw

    vec1_raw = pd.to_numeric(vec1_raw, errors='coerce').fillna(0).values.reshape(1, -1)
    vec2_raw = pd.to_numeric(vec2_raw, errors='coerce').fillna(0).values.reshape(1, -1)

    vec1_norm = normalize(vec1_raw, norm='l2', axis=1)
    vec2_norm = normalize(vec2_raw, norm='l2', axis=1)
    distance = spatial.distance.cosine(vec1_norm[0], vec2_norm[0])
    print(f'Расстояние между the beatles и coldplay: {round(distance, 2)}')
else:
    print("the beatles или coldplay отсутствуют в данных для вычисления расстояния.")


from scipy import spatial
def pClosest(points, pt, K=10):
    ind = [i[0] for i in sorted(enumerate(points), key=lambda x: spatial.distance.cosine(x[1], pt))]
    return ind[:K]

points = ratings_normalized
for cluster_id in range(5):
    centroid = centroids[cluster_id]
    closest_indices = pClosest(points, centroid)
    closest_artists = [ratings_for_clustering.index[i] for i in closest_indices]
    print(f"Кластер {cluster_id}: {closest_artists}")
    from matplotlib import pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    from sklearn.metrics import adjusted_rand_score
    import time
    import numpy as np

    X, y_true = make_moons(n_samples=100, noise=0.1, random_state=42)

    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
    plt.title("Ground truth")
    plt.show()

    # KMeans
    start = time.time()
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    time_kmeans = time.time() - start
    ari_kmeans = adjusted_rand_score(y_true, y_kmeans)

    print(f"KMeans — ARI: {ari_kmeans:.3f}, время: {time_kmeans:.5f} сек")

    # DBSCAN
    best_ari = -1
    best_params = {}
    y_dbscan_best = None
    time_dbscan_best = None

    eps_range = [0.05, 0.1, 0.2, 0.28, 0.3, 0.32]
    min_samples_range = [4, 5, 6, 7]

    for eps in eps_range:
        for min_samples in min_samples_range:
            start = time.time()
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = dbscan.fit_predict(X)
            elapsed = time.time() - start

            # Игнорируем случай, когда все точки — шум (-1)
            if len(np.unique(y_pred)) < 2:
                continue

            ari = adjusted_rand_score(y_true, y_pred)
            if ari > best_ari:
                best_ari = ari
                best_params = {'eps': eps, 'min_samples': min_samples}
                y_dbscan_best = y_pred
                time_dbscan_best = elapsed

    ari_dbscan = best_ari
    print(f"DBSCAN — лучшие параметры: {best_params}, ARI: {ari_dbscan:.3f}, время: {time_dbscan_best:.5f} сек")

    # Иерархическая кластеризация
    linkage_options = ['ward', 'complete', 'average', 'single']
    best_ari = -1
    best_linkage = None
    y_agg_best = None
    time_agg_best = None

    for linkage in linkage_options:
        # 'ward' допускает только euclidean
        if linkage == 'ward':
            metric = 'euclidean'
        else:
            metric = 'euclidean'  # по умолчанию, хотя можно и другие

        start = time.time()
        agg = AgglomerativeClustering(n_clusters=2, linkage=linkage)
        y_pred = agg.fit_predict(X)
        elapsed = time.time() - start

        ari = adjusted_rand_score(y_true, y_pred)
        if ari > best_ari:
            best_ari = ari
            best_linkage = linkage
            y_agg_best = y_pred
            time_agg_best = elapsed

    ari_agg = best_ari
    print(f"Agglomerative — лучший linkage: {best_linkage}, ARI: {ari_agg:.3f}, время: {time_agg_best:.5f} сек")

    # Спектральная кластеризация
    best_ari = -1
    best_n_neighbors = None
    y_spec_best = None
    time_spec_best = None

    for n_neighbors in range(1, 20):
        try:
            start = time.time()
            spec = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                                      n_neighbors=n_neighbors, random_state=42)
            y_pred = spec.fit_predict(X)
            elapsed = time.time() - start

            ari = adjusted_rand_score(y_true, y_pred)
            if ari > best_ari:
                best_ari = ari
                best_n_neighbors = n_neighbors
                y_spec_best = y_pred
                time_spec_best = elapsed
        except Exception as e:
            # Иногда выбрасывает ошибки при слишком малом n_neighbors
            continue

    ari_spec = best_ari
    print(f"Spectral — лучший n_neighbors: {best_n_neighbors}, ARI: {ari_spec:.3f}, время: {time_spec_best:.5f} сек")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    methods = [
        ("KMeans", y_kmeans, time_kmeans, ari_kmeans),
        ("DBSCAN", y_dbscan_best, time_dbscan_best, ari_dbscan),
        ("Agglomerative", y_agg_best, time_agg_best, ari_agg),
        ("Spectral", y_spec_best, time_spec_best, ari_spec),
        ("Ground truth", y_true, 0, 1.0)
    ]

    axes = axes.flatten()
    for i, (name, y_pred, t, ari) in enumerate(methods):
        axes[i].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
        axes[i].set_title(f"{name}\nARI: {ari:.3f}, время: {t:.5f} сек")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Уберём лишнюю подграфику
    axes[-1].remove()

    plt.tight_layout()
    plt.show()