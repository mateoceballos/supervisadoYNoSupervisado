import numpy as np
from sklearn.cluster import KMeans

# Datos de entrenamiento (Horas de estudio y calificaciones)
data = np.array([[2, 65],   # Estudiante E1
                 [5, 70],   # Estudiante E2
                 [8, 85],   # Estudiante E3
                 [10, 90],  # Estudiante E4
                 [1, 50],   # Estudiante E5
                 [7, 80]])  # Estudiante E6

# Crear el modelo K-means con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
# Al ajustar el numero de clousters realiza la predicción de 
# Ajustar el modelo a los datos
kmeans.fit(data)

# Obtener las etiquetas de los clusters y los centroides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Mostrar los resultados
print("Etiquetas de los clusters:", labels)
print("Centroides de los clusters:", centroids)