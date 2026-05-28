import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
df = pd.read_excel(r"C:\Users\ummek\Downloads\POSTDOC\DataAnalysis\data_for_cluster.xlsx", sheet_name='seedling_data')
#print(df)
df = df.fillna(7)
#print(df.columns.tolist())
features = [
    'Seed Mass(mg)', 'Seed Diameter (mm)', 'day_radicle emergence', 'day_primary_root', 'day_roothair_primary_root', 
    'Plumule Emergence_day', 'day_secondary_root', 'day_roothair_secondary_root', 'day_tertiary_root', 'root_length_day6 (mm)', 'root_dia_day6 (mm)'
]
df5 = pd.read_excel(r"C:\Users\ummek\python_practice\okra_veg.xlsx", sheet_name='Sheet5')
df6 = pd.read_excel(r"C:\Users\ummek\python_practice\okra_veg.xlsx", sheet_name='Sheet6')

df_env = pd.concat([df5, df6], ignore_index=True)
df_env.columns = df_env.columns.str.strip()

env_features = [
    'Temp (F)', 'CO2 (ppm)', 'Relative Humidity (%)',
    'Luminous Flux (lux)', 'Soil Temperature (F)',
    'Soil PH', 'Soil moisture content (%)'
]

# Average environmental conditions per Subject
df_env_summary = (
    df_env
    .groupby('Subject')[env_features]
    .mean()
    .reset_index()
)
df_combined = df.merge(df_env_summary, on='Subject', how='inner')
all_features = features + env_features
X = df_combined[all_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = KernelPCA(n_components=2,
    kernel='rbf',   # radial basis = nonlinear
    gamma=0.2,      # controls nonlinearity
    random_state=42)
X_pca = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=3, random_state=42)
df_combined['Cluster'] = kmeans.fit_predict(X_pca)
print(df_combined[['Subject', 'Cluster']])
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1],
            c=df_combined['Cluster'], cmap='viridis', s=60)

for i, subj in enumerate(df_combined['Subject']):
    plt.text(X_pca[i,0]+0.02, X_pca[i,1]+0.02, subj, fontsize=8)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Plant Clusters based on Seedlings and Environmental Features')
plt.colorbar(label='Cluster')
plt.show()
cluster_summary = df_combined.groupby("Cluster").mean(numeric_only=True)
pd.set_option("display.max_columns", None)
print(cluster_summary)

# =====================================
# FINAL growth data
# =====================================
df_growth5 = pd.read_excel(r"C:\Users\ummek\python_practice\okra_veg.xlsx", sheet_name='Sheet5')
df_growth6 = pd.read_excel(r"C:\Users\ummek\python_practice\okra_veg.xlsx", sheet_name='Sheet6')

# Strip any whitespace from column names
df_growth5.columns = df_growth5.columns.str.strip()
df_growth6.columns = df_growth6.columns.str.strip()

# Keep only relevant columns
growth_features = ['Subject', 'Plant height(mm)', 'Stem Diameter(mm)']
df_growth5 = df_growth5[growth_features]
df_growth6 = df_growth6[growth_features]

# Combine both sheets into a single dataframe
df_growth = pd.concat([df_growth5, df_growth6], ignore_index=True)

df_growth = df_growth[['Subject', 'Plant height(mm)', 'Stem Diameter(mm)']]

# =====================================
# Merge early vigor with final growth
# =====================================
df_merged = df_combined.merge(
    df_growth,
    on='Subject',
    how='left'
)

# =====================================
# Descriptive statistics
# =====================================
print("\nFinal growth by early vigor cluster:")
print(
    df_merged
    .groupby('Cluster')[['Plant height(mm)', 'Stem Diameter(mm)']]
    .mean()
)

# =====================================
# BOXPLOTS
# =====================================

# ---- Final Height ----
print((df_merged[df_merged['Cluster'] == 1]['Plant height(mm)'] == 0).sum())
plt.figure(figsize=(7,5))

height_data = [
    df_merged[df_merged['Cluster'] == c]['Plant height(mm)'].dropna()
    for c in sorted(df_merged['Cluster'].unique())
]

plt.boxplot(height_data, labels=sorted(df_merged['Cluster'].unique()),showfliers=False)
plt.xlabel('Plant cluster based on early seedling traits and environmental features')
plt.ylabel('Final plant height (mm)')
plt.title('Does early vigor and environmental features translate into final height?')
plt.tight_layout()
plt.show()

# ---- Stem Diameter ----
plt.figure(figsize=(7,5))

stem_data = [
    df_merged[df_merged['Cluster'] == c]['Stem Diameter(mm)'].dropna()
    for c in sorted(df_merged['Cluster'].unique())
]

plt.boxplot(stem_data, labels=sorted(df_merged['Cluster'].unique()),showfliers=False)
plt.xlabel('Plant cluster based on early seedling traits and environmental features')
plt.ylabel('Final stem diameter (mm)')
plt.title('Does early vigor and environmental features translate into stem thickness?')
plt.tight_layout()
plt.show()

# =====================================
# summary
# =====================================
print("\nDifference between highest and lowest early vigor groups:")

height_means = df_merged.groupby('Cluster')['Plant height(mm)'].mean()
stem_means = df_merged.groupby('Cluster')['Stem Diameter(mm)'].mean()

print("Height difference (max - min):",
      height_means.max() - height_means.min())

print("Stem diameter difference (max - min):",
      stem_means.max() - stem_means.min())
