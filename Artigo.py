import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re

# Carregando o CSV
df = pd.read_csv('dados.csv', sep=';')

# Removendo caracteres não numéricos e substituindo vírgulas por pontos na coluna 'Quantidade'
df['Quantidade'] = df['Quantidade'].apply(lambda x: re.sub(r'[^\d.]', '', x)).str.replace(',', '.').astype(float)

# Selecionando as colunas relevantes
data = df[['Quantidade', 'Ano', 'Razão Social', 'Estado', 'Município', 'Poluente emitido']]

# Normalizando os dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Quantidade', 'Ano']])

# Aplicando o algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(data_scaled)

# Mapeamento de cores para os clusters
cluster_colors = {0: 'purple', 1: 'green', 2: 'red'}

# Atribuindo cores e significados aos clusters
df['cluster_color'] = df['cluster'].map(cluster_colors)
df['cluster_label'] = df['cluster'].map({
    0: 'Moderada Estabilidade nos Níveis de Poluição ao Longo do Tempo',
    1: 'Baixos Níveis de Poluição e/ou Redução Significativa ao Longo do Tempo',
    2: 'Níveis Elevados de Poluição com Variações ao Longo do Tempo'
})

# Visualizando os clusters com legenda
for cluster, color in cluster_colors.items():
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['Quantidade'], cluster_data['Ano'], c=cluster_data['cluster_color'], label=f'Cluster {cluster}: {cluster_data["cluster_label"].iloc[0]}', alpha=0.7)

plt.xlabel('Quantidade')
plt.ylabel('Ano')
plt.title('Clusters de Emissão de Poluentes')
plt.legend()
plt.show()

# Criando gráficos individuais para os rankings
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Ranking das top 3 empresas que mais emitem poluição
top_empresas = df.groupby('Razão Social')['Quantidade'].sum().nlargest(3)
axes[0].bar(top_empresas.index, top_empresas.values, color='blue')
axes[0].set_title('Top 3 Empresas que mais emitem poluição')
axes[0].set_ylabel('Quantidade')

# Ranking dos top 3 poluentes mais emitidos
top_poluentes = df.groupby('Poluente emitido')['Quantidade'].sum().nlargest(3)
axes[1].bar(top_poluentes.index, top_poluentes.values, color='green')
axes[1].set_title('Top 3 Poluentes mais emitidos')
axes[1].set_ylabel('Quantidade')

# Ranking dos top 15 estados/municípios mais poluentes
top_municipios = df.groupby(['Estado', 'Município'])['Quantidade'].sum().nlargest(15)
# Converter as tuplas em strings
top_municipios.index = [f'{estado} - {municipio}' for estado, municipio in top_municipios.index]
axes[2].bar(top_municipios.index, top_municipios.values, color='orange')
axes[2].set_title('Top 15 Estados/Municípios mais poluentes')
axes[2].set_ylabel('Quantidade')
axes[2].tick_params(axis='x', rotation=45, labelsize=8)  # Ajuste de rotação e tamanho do texto no eixo x

plt.tight_layout()
plt.show()
