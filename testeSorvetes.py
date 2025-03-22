import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from azureml.core import Workspace, Experiment, Model

# Carregar os dados
df = pd.read_csv('vendas_sorvete.csv')

# Separar variáveis independentes (X) e dependentes (y)
X = df[['Temperatura']]
y = df['Vendas de Sorvete']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliação do modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R2 Score: {r2}')

# Plotar os resultados
plt.scatter(X_test, y_test, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regressão Linear')
plt.xlabel('Temperatura')
plt.ylabel('Vendas de Sorvete')
plt.legend()
plt.show()

# Conectar ao workspace do Azure ML
ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name='modelo-regressao-sorvete')

# Registrar o modelo no Azure ML
model_path = 'modelo_regressao.pkl'
pd.to_pickle(modelo, model_path)
model = Model.register(workspace=ws, model_name='modelo_regressao_sorvete', model_path=model_path)

print(f'Modelo registrado no Azure ML: {model.name} - {model.version}')
