import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
df = pd.read_excel(r"data_for_cluster.xlsx", sheet_name='seedling_data')
#print(df)
df = df.fillna(7)
#print(df.columns.tolist())
features = [
    'Seed Mass(mg)', 'Seed Diameter (mm)', 'day_radicle emergence', 'day_primary_root', 'day_roothair_primary_root', 
    'Plumule Emergence_day', 'day_secondary_root', 'day_roothair_secondary_root', 'day_tertiary_root', 'root_length_day6 (mm)', 'root_dia_day6 (mm)'
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(df[["Subject", "Cluster"]])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
components = pd.DataFrame(pca.components_, columns=X.columns, index=['PC1', 'PC2'])
print(components)
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Display results
print("Explained variance ratio per component:")
for i, ratio in enumerate(explained_variance_ratio, start=1):
    print(f"PC{i}: {ratio:.4f} ({ratio*100:.2f}%)")

print("\nCumulative explained variance:")
for i, cum_ratio in enumerate(cumulative_variance, start=1):
    print(f"PC1..PC{i}: {cum_ratio:.4f} ({cum_ratio*100:.2f}%)")
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["Cluster"], cmap="viridis")
for i, subj in enumerate(df["Subject"]):
    plt.text(X_pca[i,0] + 0.02, X_pca[i,1] + 0.02, subj, fontsize=8)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Seedling Clusters by Growth Patterns")
plt.colorbar(label="Cluster")
plt.show()
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
pd.set_option("display.max_columns", None)
print(cluster_summary)
df2=pd.read_excel(r"data_for_cluster.xlsx", sheet_name='Sheet4')
result1=df2.loc[df2['Cluster']==0,'Subject']
result2=df2.loc[df2['Cluster']==1,'Subject']
result3=df2.loc[df2['Cluster']==2,'Subject']
print(result1.tolist()) 
print(result2.tolist()) 
print(result3.tolist()) 
