#NOTES FOR LATER:
#Replace nutrient deficiency rates by poverty bar chart nutrients w/ clinically relevant & correlated variables
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

sns.set_theme()

# LOAD/MERGE DATA

# Load datasets
demographics_df = pd.read_sas('DEMO_L.xpt')
dietary_df = pd.read_sas('DR1TOT_L.xpt')
cholesterol_df = pd.read_sas('TRIGLY_L.xpt')

# Merge
df = demographics_df.merge(
    dietary_df, on = 'SEQN', how = 'left').merge(
        cholesterol_df, on = 'SEQN', how = 'left')

# Map race codes
race_mapping = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White',
                4: 'Non-Hispanic Black', 6: 'Non-Hispanic Asian', 7: 'Other/Multi-racial'}
df['Race'] = df['RIDRETH3'].map(race_mapping)




# CLEAN DATA


# -------------- print(f"Initial rows: {len(df):,}") -----------------
# Duplicate record removal
df = df.drop_duplicates(subset = 'SEQN')

# Remove negative values from variables of interest
for col in ['INDFMPIR', 'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 
            'DR1TFIBE', 'DR1TTFAT', 'DR1TSFAT', 'LBDLDLN', 'LBXTLG']:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

# Remove rows missing all essential variables
df = df.dropna(subset = ['RIDAGEYR', 'RIAGENDR', 'RIDRETH3'], how = 'any')

# ------------ print(f"After cleaning: {len(df):,} rows") ---------------




# # ============================================================================
# # PLOT 1: LDL RANGES BY RACE
# # ============================================================================
# df['LDL_category'] = pd.cut(df['LBDLDLN'], bins = [0, 150, 200, np.inf],
#                             labels=['Normal (<150)', 'Borderline (150-199)', 'High (≥200)'])

# fig, ax = plt.subplots(figsize=(14, 8))

# # Calculate percentages
# race_ldl = df.groupby(['Race', 'LDL_category'], observed = True).size().reset_index(name = 'count')
# race_totals = df.groupby('Race').size().reset_index(name = 'total')
# race_ldl = race_ldl.merge(race_totals, on = 'Race')
# race_ldl['percentage'] = (race_ldl['count'] / race_ldl['total']) * 100

# # Plot
# races = ['Mexican American', 'Other Hispanic', 'Non-Hispanic White', 
#          'Non-Hispanic Black', 'Non-Hispanic Asian', 'Other/Multi-racial']
# tgly_categories = ['Normal (<150)', 'Borderline (150-199)', 'High (≥200)']
# x = np.arange(len(races))
# width = 0.25
# colors_tgly = ["#66c94b", "#fcc25f", "#e94e4e"]

# for i, category in enumerate(tgly_categories):
#     category_data = race_ldl[race_ldl['LDL_category'] == category]
    
#     # Get percentages for each race (0 if race not in data)
#     percentages = []
#     for race in races:
#         race_data = category_data[category_data['Race'] == race]
#         if len(race_data) > 0:
#             percentages.append(race_data['percentage'].values[0])
#         else:
#             percentages.append(0)
    
#     # Create bars
#     bars = ax.bar(x + i * width, percentages, width, label = category, 
#                   color = colors_tgly[i], alpha = 0.8, edgecolor = 'black', linewidth = 0.5)
    
#     # Add percentage labels on bars
#     for j, bar in enumerate(bars):
#         height = bar.get_height()
#         if height > 0:
#             ax.text(bar.get_x() + bar.get_width()/2., height + 1,
#                    f'{height:.1f}%',
#                    ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold')

# ax.set_xlabel('Race', fontsize = 13, fontweight = 'bold')
# ax.set_ylabel('Percentage (%)', fontsize = 13, fontweight = 'bold')
# ax.set_title('High-LDL Risk Categories by Race', fontsize=15, fontweight='bold', pad=20)
# ax.set_xticks(x + width)
# ax.set_xticklabels(races, rotation=45, ha='right')
# ax.legend(title='LDL Level', loc='upper right')
# ax.set_ylim(0, 100)
# ax.grid(axis='y', alpha=0.3)

# plt.tight_layout()
# plt.savefig('LDL_ranges_by_race.png', dpi=300, bbox_inches='tight')
# plt.show()

# ============================================================================
# PLOT 2: NUTRIENT DEFICIENCY BY POVERTY
# A grouped bar chart showing how often people in different income brackets fall below the 
# recommended daily intake for key nutrients. 
# Income categories are based on the poverty-to-income ratio (PIR)
# ============================================================================

df['Poverty_Category'] = pd.cut(df['INDFMPIR'], bins = [0, 1, 2, 3, 5],
                                labels = ['<1.0\n(Low-income)', '1.0-2.0\n(Near Low-income)', 
                                       '2.0-3.0\n(Middle)', '>3.0\n(High-income)'])

fig, ax = plt.subplots(figsize = (14, 8))

# RDA thresholds
rda = {'DR1TFIBE': 25, 'DR1TVC': 75, 'DR1TCALC': 1000, 'DR1TIRON': 18, 'DR1TVD': 15}

# Calculate deficiency rates
deficiency_data = []
for nutrient, threshold in rda.items():
    for category in df['Poverty_Category'].dropna().unique():
        subset = df[df['Poverty_Category'] == category]
        rate = (subset[nutrient] < threshold).sum() / subset[nutrient].notna().sum() * 100 if subset[nutrient].notna().sum() > 0 else 0
        deficiency_data.append({'Nutrient': nutrient.replace('DR1T', '').replace('DR1', ''),
                               'Poverty Category': category, 'Rate': rate})

deficiency_df = pd.DataFrame(deficiency_data)

# Plot
nutrients = deficiency_df['Nutrient'].unique()
poverty_categories = ['<1.0\n(Low-income)', '1.0-2.0\n(Near Low-income)', 
                      '2.0-3.0\n(Middle)', '>3.0\n(High-income)']
x = np.arange(len(nutrients))
width = 0.2
colors = ["#eb6868", "#ffa277", "#ffe681", "#d4ed62"]

for i, category in enumerate(poverty_categories):
    rates = [deficiency_df[(deficiency_df['Nutrient'] == n) & 
                           (deficiency_df['Poverty Category'] == category)]['Rate'].values[0]
             if len(deficiency_df[(deficiency_df['Nutrient'] == n) & 
                                  (deficiency_df['Poverty Category'] == category)]) > 0 else 
                                  0 for n in nutrients]
    ax.bar(x + i * width, rates, width, label = category, color = colors[i], alpha = 0.8)

ax.set_xlabel('Nutrient', fontsize = 12, fontweight = 'bold')
ax.set_ylabel('Deficiency Rate (%)', fontsize = 12, fontweight = 'bold')
ax.set_title('Nutrient Deficiency Rates by Poverty Category', fontsize = 14, fontweight = 'bold', pad = 20)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(nutrients)
ax.legend(title = 'Poverty Category')
ax.grid(axis = 'y', alpha = 0.3)

plt.tight_layout()
plt.savefig('nutrient_deficiency_by_poverty.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# ============================================================================
# PLOT 3: LDL & TRIGLYCERIDE HEATMAPS
# Heatmaps showing average LDL and triglyceride levels broken down by both race and poverty category.
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize = (20, 10))

race_order = ['Non-Hispanic White', 'Non-Hispanic Black', 'Non-Hispanic Asian',
              'Mexican American', 'Other Hispanic', 'Other/Multi-racial']

# LDL Heatmap
ldl_pivot = df.groupby(['Race', 'Poverty_Category'], observed = True)['LBDLDL'].mean().reset_index().pivot(
    index = 'Race', columns = 'Poverty_Category', values = 'LBDLDL').reindex(race_order)

sns.heatmap(ldl_pivot, annot = True, fmt = '.1f', cmap = 'RdYlGn_r', ax = axes[0],
            vmin = 80, vmax = 160, center = 115, linewidths = 1, linecolor = 'white',
            cbar_kws = {'label': 'Mean LDL (mg/dL)'})
axes[0].set_title('Mean LDL by Race and Poverty', fontsize = 13, fontweight = 'bold', pad = 15)
axes[0].set_xlabel('Poverty Ratio', fontsize = 11, fontweight = 'bold')
axes[0].set_ylabel('Race', fontsize = 11, fontweight = 'bold')

# Triglyceride Heatmap
trig_pivot = df.groupby(['Race', 'Poverty_Category'], observed = True)['LBXTLG'].mean().reset_index().pivot(
    index = 'Race', columns = 'Poverty_Category', values = 'LBXTLG').reindex(race_order)

sns.heatmap(trig_pivot, annot = True, fmt = '.1f', cmap = 'RdYlGn_r', ax = axes[1],
            vmin = 50, vmax = 200, center = 125, linewidths = 1, linecolor = 'white',
            cbar_kws = {'label': 'Mean Triglycerides (mg/dL)'})
axes[1].set_title('Mean Triglycerides by Race and Poverty', fontsize=13, fontweight = 'bold', pad=15)
axes[1].set_xlabel('Poverty Ratio', fontsize = 11, fontweight = 'bold')
axes[1].set_ylabel('Race', fontsize = 11, fontweight = 'bold')

fig.suptitle('LDL and Triglyceride Levels by Race and Poverty', fontsize = 15, fontweight = 'bold', y = 0.98)
plt.tight_layout()
plt.savefig('ldl_triglyceride_heatmaps.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# ============================================================================
# PLOT 4: CORRELATION BAR CHART
# Bar chart ranking how strongly various dietary and demographic variables correlate with LDL cholesterol,
# representing both negative and positive correlation.
# ============================================================================

# Select variables
vars_dict = {'RIDAGEYR': 'Age', 'INDFMPIR': 'Poverty Ratio', 'DR1TKCAL': 'Calories',
             'DR1TPROT': 'Protein', 'DR1TCARB': 'Carbs', 'DR1TSUGR': 'Sugars',
             'DR1TFIBE': 'Fiber', 'DR1TTFAT': 'Total Fat', 'DR1TSFAT': 'Sat Fat',
             'DR1TMFAT': 'Monounsat Fat', 'DR1TPFAT': 'Polyunsat Fat',
             'DR1TCHOL': 'Cholesterol', 'DR1TSODI': 'Sodium', 'DR1TCALC': 'Calcium',
             'DR1TIRON': 'Iron', 'DR1TVD': 'Vitamin D', 'DR1TVC': 'Vitamin C',
             'LBDLDLN': 'LDL'}

corr_df = df[list(vars_dict.keys())].rename(columns = vars_dict).dropna()
correlations = corr_df.corr()['LDL'].drop('LDL').sort_values(key = abs, ascending = True)

fig, ax = plt.subplots(figsize = (12, 8))
colors = ["#de5568" if x > 0 else "#49cce3" for x in correlations.values]
ax.barh(range(len(correlations)), correlations.values, color = colors, 
        alpha = 0.7, edgecolor = 'black', linewidth = 0.5)
ax.set_yticks(range(len(correlations)))
ax.set_yticklabels(correlations.index)
ax.set_xlabel('Correlation Coefficient', fontsize = 12, fontweight = 'bold')
ax.set_title('Nutrient & Poverty Correlations with LDL', fontsize = 14, fontweight = 'bold', pad=20)
ax.axvline(x = 0, color = 'black', linestyle = '-', linewidth = 1)
ax.grid(axis = 'x', alpha = 0.3)

plt.tight_layout()
plt.savefig('correlation_bar_chart_LDL.png', dpi = 300, bbox_inches = 'tight')
plt.show()

print("Visualizations saved")





# # ====================================================================
# # INCORPORATE LATER - KMEANS CLUSTERING FOR ANALYSIS, USING MAIN()
# # =====================================================================

# def min_max_scaling(X_train, X_test=None):
#     """MinMaxScaler to compute min and max values"""
#     scaler = MinMaxScaler()  

#     # Fit the scaler on the training data (find min and max values for each feature)
#     # Then transform the training data so each feature is scaled to the range [0, 1]                  
#     X_train_scaled = scaler.fit_transform(X_train)  
    
#     # Check if test data was provided
#     if X_test is None:    
#         # Return the scaled training data and an empty list for the test set           
#         return X_train_scaled.tolist() + [] 

#     # Use the SAME min and max values learned from the training set to scale the test data  
#     # Important: we use transform(), not fit_transform(), so the test set does not affect scaling 
#     X_test_scaled = scaler.transform(X_test)   

#     # Return both scaled datasets converted to Python lists
#     return X_train_scaled.tolist() + X_test_scaled.tolist()  

# # PROBLEM 1
# def perform_kmeans(X: list, n_clusters: int) -> list:
#     """
#     Performs K-Means clustering on data.

#     Args:
#     X: Feature data (list of lists)
#     n_clusters: Number of clusters (integer)

#     Returns:
#     list: Cluster labels for each sample as a list
#     """
#     kmeans = KMeans(n_clusters = n_clusters, random_state = 2500)
#     kmeans.fit(X)
#     return kmeans.labels_.tolist()

# # PROBLEM 2
# def perform_hierarchical(X: list, n_clusters: int) -> list:
#     """
#     Performs hierarchical clustering on data.

#     Args:
#     X: Feature data (list of lists)
#     n_clusters: Number of clusters (integer)

#     Returns:
#     list: Cluster labels for each sample as a list
#     """
#     hierarchical = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'average')
#     hierarchical.fit(X)
#     return hierarchical.labels_.tolist()

# # PROBLEM 3
# def find_optimal_clusters(X: list, clustering_option = "kmeans", max_k = 10) -> list:
#     """
#     Finds the optimal number of clusters by testing different values of k 
#     and comparing their silhouette scores.

#     Args:
#     X: Feature data (list of lists)
#     clustering_option: String indicating which clustering method to use - either "kmeans" or
#     "hierarchical" (default="kmeans")
#     max_k: Maximum number of clusters to test (integer, default=10)

#     Returns:
#     int: optimal number of clusters (the k value with the highest silhouette score)
#     """
#     best_k = 2
#     best_score = -1

#     for k in range(2, max_k + 1):
#         if clustering_option == "kmeans":
#             cluster_labels = perform_kmeans(X, k)
#         else:
#             cluster_labels = perform_hierarchical(X, k)

#         score = silhouette_score(X, cluster_labels)
#         score = float(score)
#         print(f"Silhouette Score: {score:.3f}")

#         if score > best_score:
#             best_score = score
#             best_k = k
#     return best_k

# # PROBLEM 4
# def save_clustering_results(labels, filename):
#     """
#     Saves cluster labels to a CSV file.

#     Args:
#     labels (list): Predicted cluster labels
#     filename (str): Name of the output CSV file
#     """
#     with open(filename, 'w') as file:
#         file.write("cluster_label\n")
#         for label in labels:
#             file.write(f"{label}\n")

# def main():
#     # ── 1. SELECT FEATURES FOR CLUSTERING ──────────────────────────────────────
#     # Pick dietary + cholesterol + demographic variables you want to cluster on
#     features = [
#         'RIDAGEYR',   # Age
#         'INDFMPIR',   # Poverty ratio
#         'DR1TKCAL',   # Calories
#         'DR1TPROT',   # Protein
#         'DR1TCARB',   # Carbs
#         'DR1TSUGR',   # Sugars
#         'DR1TFIBE',   # Fiber
#         'DR1TTFAT',   # Total fat
#         'DR1TSFAT',   # Saturated fat
#         'LBDLDLN',    # LDL
#         'LBXTLG',     # Triglycerides
#     ]

#     # ── 2. PREP THE DATA ────────────────────────────────────────────────────────
#     cluster_df = df[features + ['Race', 'Poverty_Category']].dropna(subset=features)
#     X = cluster_df[features].values.tolist()

#     # ── 3. SCALE WITH MINMAXSCALER ──────────────────────────────────────────────
#     # All features on [0,1] so no single variable dominates (e.g. calories vs fiber)
#     X_scaled = min_max_scaling(X)

#     # ── 4. FIND OPTIMAL K (BOTH METHODS) ────────────────────────────────────────
#     print("--- KMeans silhouette scores ---")
#     kmeans_best_k = find_optimal_clusters(X_scaled, clustering_option="kmeans", max_k=8)
#     print(f"Optimal KMeans k: {kmeans_best_k}\n")

#     print("--- Hierarchical silhouette scores ---")
#     hier_best_k = find_optimal_clusters(X_scaled, clustering_option="hierarchical", max_k=8)
#     print(f"Optimal Hierarchical k: {hier_best_k}\n")

#     # ── 5. CLUSTER WITH OPTIMAL K ────────────────────────────────────────────────
#     kmeans_labels = perform_kmeans(X_scaled, n_clusters=kmeans_best_k)
#     hier_labels = perform_hierarchical(X_scaled, n_clusters=hier_best_k)

#     # ── 6. ATTACH LABELS BACK TO DATAFRAME ──────────────────────────────────────
#     cluster_df = cluster_df.copy()
#     cluster_df['KMeans_Cluster']       = kmeans_labels
#     cluster_df['Hierarchical_Cluster'] = hier_labels

#     # ── 8. ANALYZE CLUSTERS ──────────────────────────────────────────────────────
#     print("=== KMeans Cluster Profiles ===")
#     print(cluster_df.groupby('KMeans_Cluster')[features].mean().round(2).to_string())

#     print("\n=== KMeans Cluster | Race Breakdown ===")
#     print("Cluster 0: Low-income | Cluster 1: high=income")
#     print(pd.crosstab(cluster_df['KMeans_Cluster'], cluster_df['Race'], normalize='index').round(2).to_string())

#     print("\n=== KMeans Cluster | Poverty Breakdown ===")
#     print(pd.crosstab(cluster_df['KMeans_Cluster'], cluster_df['Poverty_Category'], normalize='index').round(2).to_string())
#     print("!! Inference !! : Cluster 0 is largely represented by low-income, while Cluster 1 is largely those who are not. Low-income cluster has higher triglycerides despite lower caloric intake, which aligns with research linking poverty-associated diets (refined carbs, sugar) to elevated triglycerides")
# if __name__ == '__main__':
#     main()