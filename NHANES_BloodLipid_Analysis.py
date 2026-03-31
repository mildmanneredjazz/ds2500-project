#NOTES FOR LATER:
#Replace nutrient deficiency rates by poverty bar chart nutrients w/ clinically relevant & correlated variables
#Make all code run from main()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

sns.set_theme()

# LOAD/MERGE DATA

# Load datasets
demographics_df = pd.read_sas('ds2500-project/NHANES_data/DEMO_L.xpt')
dietary_df = pd.read_sas('ds2500-project/NHANES_data/DR1TOT_L.xpt')
cholesterol_df = pd.read_sas('ds2500-project/NHANES_data/TRIGLY_L.xpt')

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




# ============================================================================
# PLOT 1: NUTRIENT DEFICIENCY BY POVERTY
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
plt.savefig('ds2500-project/Visualizations/nutrient_deficiency_by_poverty.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# ============================================================================
# PLOT 2: LDL & TRIGLYCERIDE HEATMAPS
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
plt.savefig('ds2500-project/Visualizations/ldl_triglyceride_heatmaps.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# ============================================================================
# PLOT 3: CORRELATION BAR CHART
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
plt.savefig('ds2500-project/Visualizations/correlation_bar_chart_LDL.png', dpi = 300, bbox_inches = 'tight')
plt.show()

print("Visualizations saved")






# LINEAR REGRESSION
def run_linear_regression(df):

    features = ['RIDAGEYR', 'INDFMPIR', 'DR1TSFAT', 'DR1TFIBE',
                'DR1TSUGR', 'DR1TTFAT', 'DR1TPROT', 'DR1TCARB', 'RIAGENDR']

    model_df = df[features + ['LBXTLG', 'LBDLDLN']].dropna()
    X      = model_df[features]
    y_trig = model_df['LBXTLG']
    y_ldl  = model_df['LBDLDLN']

    # Fit both models
    trig_model = LinearRegression().fit(X, y_trig)
    ldl_model  = LinearRegression().fit(X, y_ldl)

    # Print coefficients
    for name, model in [('Triglycerides', trig_model), ('LDL', ldl_model)]:
        print(f"\n--- {name} Model ---")
        for feat, coef in zip(features, model.coef_):
            print(f"  {feat}: {coef:.4f}")
        print(f"  R²: {model.score(X, model.predict(X)):.3f}")

    # Train/test performance on triglycerides
    X_train, X_test, y_train, y_test = train_test_split(X, y_trig, test_size=0.2, random_state=42)
    y_pred = LinearRegression().fit(X_train, y_train).predict(X_test)
    print(f"\nTriglyceride Test R²:   {r2_score(y_test, y_pred):.3f}")
    print(f"Triglyceride Test RMSE: {mean_squared_error(y_test, y_pred)**.5:.2f} mg/dL")

    # Coefficient plot
    coef_df = pd.DataFrame({
        'Feature':     features,
        'Coefficient': trig_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)

    colors = ['#de5568' if c > 0 else '#49cce3' for c in coef_df['Coefficient']]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Coefficient (effect on Triglycerides)')
    ax.set_title('Linear Regression Coefficients — Triglyceride Model')
    plt.tight_layout()
    plt.savefig('regression_coefficients.png', dpi=300, bbox_inches='tight')
    plt.show()





def regression_summary(df):
    features = ['RIDAGEYR', 'RIAGENDR', 'INDFMPIR', 
                'DR1TSFAT', 'DR1TFIBE', 'DR1TSUGR', 'DR1TTFAT']
    
    for target, name in [('LBXTLG', 'Triglycerides'), ('LBDLDLN', 'LDL')]:
        model_df = df[features + [target]].dropna()
        X = model_df[features]
        y = model_df[target]
        
        model = LinearRegression().fit(X, y)
        print(f"\n=== {name} ===")
        print(f"R²: {model.score(X, y):.3f}")
        for feat, coef in sorted(zip(features, model.coef_), 
                                  key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feat}: {coef:.4f}")



# CORRELATION MATRIX

def plot_correlation_matrix(df):
    vars = {'RIDAGEYR': 'Age', 'RIAGENDR': 'Gender', 'INDFMPIR': 'Poverty',
            'DR1TSFAT': 'Sat Fat', 'DR1TFIBE': 'Fiber', 'DR1TSUGR': 'Sugar',
            'DR1TTFAT': 'Total Fat', 'LBXTLG': 'Triglycerides', 'LBDLDLN': 'LDL'}
    
    corr = df[list(vars.keys())].rename(columns=vars).corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, linewidths=0.5, ax=ax)
    ax.set_title('Full Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ds2500-project/Visualizations/correlation_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # ──── LINEAR REGRESSION ────────────────
    run_linear_regression(df)
    regression_summary(df)
    plot_correlation_matrix(df)


if __name__ == '__main__':
    main()