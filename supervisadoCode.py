from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Datos de entrenamiento (Características: Edad, Peso; Etiqueta: Enfermedad)
X = np.array([[25, 85], [30, 90], [28, 91], [31, 81], [45, 80], [48, 82], [25, 88], [35, 75], [45, 95], [20, 59], [25, 55], [26, 60], [60, 57]])
y = np.array(['No','No','Sí','Sí','No','Sí','No','No','Sí','Sí','Sí','No','Sí'])
# Menores de 60k estan enfermos
# Entre 20 y 30 Años, pueden estar sanos hasta los 90k
# Entre 31 Años en adelante, No pueden superar los 80k
# Crear el modelo de árbol de decisión
modelo = DecisionTreeClassifier()

# Entrenar el modelo
modelo.fit(X, y)

# Predecir la clase de un nuevo paciente
nuevo_paciente = np.array([[35, 60]])
prediccion = modelo.predict(nuevo_paciente)
print(f'El paciente de 32 años y 85 kg tiene enfermedad: {prediccion[0]}')