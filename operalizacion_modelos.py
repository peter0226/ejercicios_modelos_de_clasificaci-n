import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelos
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)

tree = DecisionTreeClassifier()
tree.fit(X_train_scaled, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred_logreg = logreg.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test_scaled)
y_pred_tree = tree.predict(X_test_scaled)
y_pred_knn = knn.predict(X_test_scaled)
y_pred_mlp = mlp.predict(X_test_scaled)

# Calcular métricas de evaluación
print("Regresión Logística:")
print("Exactitud:", accuracy_score(y_test, y_pred_logreg))
print("Precisión:", precision_score(y_test, y_pred_logreg))
print("Exhaustividad:", recall_score(y_test, y_pred_logreg))
print("Puntuación F1:", f1_score(y_test, y_pred_logreg))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_logreg))
y_prob_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
print("AUC:", roc_auc_score(y_test, y_prob_logreg))

print("\nBosques Aleatorios:")
print("Exactitud:", accuracy_score(y_test, y_pred_rf))
print("Precisión:", precision_score(y_test, y_pred_rf))
print("Exhaustividad:", recall_score(y_test, y_pred_rf))
print("Puntuación F1:", f1_score(y_test, y_pred_rf))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_rf))
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
print("AUC:", roc_auc_score(y_test, y_prob_rf))

print("\nÁrbol de Decisión:")
print("Exactitud:", accuracy_score(y_test, y_pred_tree))
print("Precisión:", precision_score(y_test, y_pred_tree))
print("Exhaustividad:", recall_score(y_test, y_pred_tree))
print("Puntuación F1:", f1_score(y_test, y_pred_tree))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_tree))
y_prob_tree = tree.predict_proba(X_test_scaled)[:, 1]
print("AUC:", roc_auc_score(y_test, y_prob_tree))

print("\nK-Vecinos más Cercanos:")
print("Exactitud:", accuracy_score(y_test, y_pred_knn))
print("Precisión:", precision_score(y_test, y_pred_knn))
print("Exhaustividad:", recall_score(y_test, y_pred_knn))
print("Puntuación F1:", f1_score(y_test, y_pred_knn))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_knn))
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
print("AUC:", roc_auc_score(y_test, y_prob_knn))

print("\nRed Neuronal:")
print("Exactitud:", accuracy_score(y_test, y_pred_mlp))
print("Precisión:", precision_score(y_test, y_pred_mlp))
print("Exhaustividad:", recall_score(y_test, y_pred_mlp))
print("Puntuación F1:", f1_score(y_test, y_pred_mlp))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_mlp))
y_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
print("AUC:", roc_auc_score(y_test, y_prob_mlp))

# Visualizaciones
# Curva ROC
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
roc_auc_tree = auc(fpr_tree, tpr_tree)

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

plt.figure()
plt.plot(fpr_logreg, tpr_logreg, label='Regresión Logística (AUC = %0.2f)' % roc_auc_logreg)
plt.plot(fpr_rf, tpr_rf, label='Bosques Aleatorios (AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_tree, tpr_tree, label='Árbol de Decisión (AUC = %0.2f)' % roc_auc_tree)
plt.plot(fpr_knn, tpr_knn, label='K-Vecinos más Cercanos (AUC = %0.2f)' % roc_auc_knn)
plt.plot(fpr_mlp, tpr_mlp, label='Red Neuronal (AUC = %0.2f)' % roc_auc_mlp)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Matriz de Confusión (para todos los modelos)
modelos = [("Regresión Logística", y_pred_logreg), ("Bosques Aleatorios", y_pred_rf), ("Arbol De Decision", y_pred_tree), ("K-Vecinos", y_pred_knn), ("Red Neuronal", y_pred_mlp)]
for nombre, predicciones in modelos:
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, predicciones), annot=True, fmt='d', cmap='viridis')
    plt.title(f'Matriz de Confusión - {nombre}')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.show()

# Gráfico de barras de métricas (para todos los modelos)
metricas = ['Exactitud', 'Precisión', 'Exhaustividad', 'Puntuación F1', 'AUC']
valores_modelos = {
    "Regresión Logística": [accuracy_score(y_test, y_pred_logreg), precision_score(y_test, y_pred_logreg), recall_score(y_test, y_pred_logreg), f1_score(y_test, y_pred_logreg), roc_auc_score(y_test, y_prob_logreg)],
    "Bosques Aleatorios": [accuracy_score(y_test, y_pred_rf), precision_score(y_test, y_pred_rf), recall_score(y_test, y_pred_rf), f1_score(y_test, y_pred_rf), roc_auc_score(y_test, y_prob_rf)],
    "Arbol De Decision": [accuracy_score(y_test, y_pred_tree), precision_score(y_test, y_pred_tree), recall_score(y_test, y_pred_tree), f1_score(y_test, y_pred_tree), roc_auc_score(y_test, y_prob_tree)],
    "K-Vecinos": [accuracy_score(y_test, y_pred_knn), precision_score(y_test, y_pred_knn), recall_score(y_test, y_pred_knn), f1_score(y_test, y_pred_knn), roc_auc_score(y_test, y_prob_knn)],
    "Red Neuronal": [accuracy_score(y_test, y_pred_mlp), precision_score(y_test, y_pred_mlp), recall_score(y_test, y_pred_mlp), f1_score(y_test, y_pred_mlp), roc_auc_score(y_test, y_prob_mlp)],
}

x = np.arange(len(metricas))
width = 0.15

plt.figure(figsize=(12, 6))
for i, (nombre, valores) in enumerate(valores_modelos.items()):
    plt.bar(x + (i * width), valores, width, label=nombre)

plt.xticks(x, metricas)
plt.ylabel('Puntuación')
plt.title('Comparación de Métricas de Evaluación')
plt.legend()
plt.show()