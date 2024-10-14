import numpy as np
from sklearn.cluster import KMeans

# Datos de entrenamiento (Horas de estudio y calificaciones)
data = np.array([[2, 65],   # Estudiante E1
                 [5, 70],   # Estudiante E2
                 [8, 85],   # Estudiante E3
                 [10, 90],  # Estudiante E4
                 [1, 50],   # Estudiante E5
                 [7, 80]])  # Estudiante E6

# Creamos el modelo K-means con 2 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
# Al ajustar el numero de clousters realiza la predicción de dichos clousters

# Ajustamos el modelo a los datos
kmeans.fit(data)

# Obtenemos las etiquetas de los clusters y los centroides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Mostramos los resultados
print("Etiquetas de los clusters:", labels)
print("Centroides de los clusters:", centroids)