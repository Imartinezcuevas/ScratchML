import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import os
import sys
# Agrega la carpeta raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from supervisedLearning.knn import KNN

def plot_knn_results(X_train, y_train, X_test, y_test, y_pred):
    y_pred = np.reshape(y_pred, (-1,))
    # Colores para las tres clases
    colors = ['purple', 'green', 'orange']
    
    plt.figure(figsize=(10, 6))
    
    # Graficamos los puntos de entrenamiento con colores según sus clases
    for class_value in np.unique(y_train):
        X_class = X_train[y_train == class_value]
        plt.scatter(X_class[:, 0], X_class[:, 1], 
                    color=colors[class_value], label=f'Train Class {class_value}', marker='s', edgecolor='k', s=50, alpha=0.6)

    # Graficamos los puntos de prueba con colores según las clases predichas
    for class_value in np.unique(y_pred):
        X_class = X_test[y_pred == class_value]
        plt.scatter(X_class[:, 0], X_class[:, 1], 
                    color=colors[class_value], label=f'Test Pred Class {class_value}', edgecolor='k', s=50, marker='o')
    
    # Agregamos contornos para indicar las clases reales de los puntos de prueba y marcamos errores
    for i in range(len(X_test)):
        plt.scatter(X_test[i, 0], X_test[i, 1], 
                    edgecolor='red' if y_test[i] != y_pred[i] else 'black', 
                    facecolor='none', s=80, linewidth=1.5, marker='o')
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("KNN Classification Results with PCA (Training and Test Data)")
    plt.legend(loc='best')
    plt.show()



def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test, X_train_pca, X_test_pca = train_test_split(X, y, X_pca, test_size=0.3)

    model = KNN(n_neighbors=5)
    y_pred = model.predict(X_train, y_train, X_test)

    accuracy= accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    # Llamamos a la función de graficación
    plot_knn_results(X_train_pca, y_train, X_test_pca, y_test, y_pred)

if __name__ == "__main__":
    main()

