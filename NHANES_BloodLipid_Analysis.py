#NOTES FOR LATER:
#Replace nutrient deficiency rates by poverty bar chart nutrients w/ clinically relevant & correlated variables
#TODO: replace VC and VD with sugar, focus on poverty & nutrition/obesity

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

sns.set_theme()

# ============================================================================
# LOAD & MERGE
# ============================================================================

demographics_df = pd.read_sas('ds2500-project/NHANES_data/DEMO_L.xpt')
dietary_df      = pd.read_sas('ds2500-project/NHANES_data/DR1TOT_L.xpt')
cholesterol_df  = pd.read_sas('ds2500-project/NHANES_data/TRIGLY_L.xpt')

df = demographics_df.merge(dietary_df, on='SEQN', how='left').merge(cholesterol_df, on='SEQN', how='left')

race_mapping = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White',
                4: 'Non-Hispanic Black', 6: 'Non-Hispanic Asian', 7: 'Other/Multi-racial'}
df['Race'] = df['RIDRETH3'].map(race_mapping)

# ============================================================================
# CLEAN
# ============================================================================


for col in ['INDFMPIR', 'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR',
            'DR1TFIBE', 'DR1TTFAT', 'DR1TSFAT', 'LBDLDLN', 'LBXTLG']:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

df = df.dropna(subset=['RIDAGEYR', 'RIAGENDR', 'RIDRETH3'])

df['Poverty_Category'] = pd.cut(df['INDFMPIR'], bins=[0, 1, 2, 3, 5],
                                labels=['<1.0 (Low)', '1.0-2.0 (Near Low)', '2.0-3.0 (Middle)', '>3.0 (High)'])

# ============================================================================
# PLOT 1: NUTRIENT DEFICIENCY BY POVERTY
# ============================================================================

def plot_nutrient_deficiency(df):
    rda = {'DR1TFIBE': 25, 'DR1TVC': 75, 'DR1TCALC': 1000, 'DR1TIRON': 18, 'DR1TVD': 15}
    poverty_categories = ['<1.0 (Low)', '1.0-2.0 (Near Low)', '2.0-3.0 (Middle)', '>3.0 (High)']
    colors = ["#eb6868", "#ffa277", "#ffe681", "#d4ed62"]

    deficiency_data = []
    for nutrient, threshold in rda.items():
        for category in poverty_categories:
            subset = df[df['Poverty_Category'] == category]
            rate = (subset[nutrient] < threshold).sum() / subset[nutrient].notna().sum() * 100
            deficiency_data.append({'Nutrient': nutrient.replace('DR1T', ''), 
                                    'Poverty Category': category, 'Rate': rate})

    deficiency_df = pd.DataFrame(deficiency_data)
    nutrients = deficiency_df['Nutrient'].unique()
    x = np.arange(len(nutrients))

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, category in enumerate(poverty_categories):
        rates = [deficiency_df[(deficiency_df['Nutrient'] == n) & 
                               (deficiency_df['Poverty Category'] == category)]['Rate'].values[0] for n in nutrients]
        ax.bar(x + i * 0.2, rates, 0.2, label=category, color=colors[i], alpha=0.8)

    ax.set_xticks(x + 0.3)
    ax.set_xticklabels(nutrients)
    ax.set_xlabel('Nutrient', fontweight='bold')
    ax.set_ylabel('Deficiency Rate (%)', fontweight='bold')
    ax.set_title('Nutrient Deficiency Rates by Poverty Category', fontweight='bold')
    ax.legend(title='Poverty Category')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('ds2500-project/Visualizations/nutrient_deficiency_by_poverty.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# PLOT 2: LDL & TRIGLYCERIDE HEATMAPS
# ============================================================================

def plot_heatmaps(df):
    race_order = ['Non-Hispanic White', 'Non-Hispanic Black', 'Non-Hispanic Asian',
                  'Mexican American', 'Other Hispanic', 'Other/Multi-racial']

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, col, label, vmin, vmax, center in [
        (axes[0], 'LBDLDL',  'Mean LDL (mg/dL)',          80,  160, 115),
        (axes[1], 'LBXTLG',  'Mean Triglycerides (mg/dL)', 50,  200, 125)]:

        pivot = df.groupby(['Race', 'Poverty_Category'], observed=True)[col].mean().reset_index().pivot(
            index='Race', columns='Poverty_Category', values=col).reindex(race_order)

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax,
                    vmin=vmin, vmax=vmax, center=center, linewidths=1, linecolor='white',
                    cbar_kws={'label': label})
        ax.set_title(f'{label} by Race and Poverty', fontweight='bold')

    fig.suptitle('LDL and Triglyceride Levels by Race and Poverty', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ds2500-project/Visualizations/ldl_triglyceride_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# PLOT 3: CORRELATION BAR CHART
# ============================================================================

def plot_correlation_bar(df):
    vars_dict = {'RIDAGEYR': 'Age', 'INDFMPIR': 'Poverty Ratio', 'DR1TKCAL': 'Calories',
                 'DR1TPROT': 'Protein', 'DR1TCARB': 'Carbs', 'DR1TSUGR': 'Sugars',
                 'DR1TFIBE': 'Fiber', 'DR1TTFAT': 'Total Fat', 'DR1TSFAT': 'Sat Fat',
                 'DR1TMFAT': 'Monounsat Fat', 'DR1TPFAT': 'Polyunsat Fat',
                 'DR1TCHOL': 'Cholesterol', 'DR1TSODI': 'Sodium', 'DR1TCALC': 'Calcium',
                 'DR1TIRON': 'Iron', 'DR1TVD': 'Vitamin D', 'DR1TVC': 'Vitamin C', 'LBDLDLN': 'LDL'}

    corr_df = df[list(vars_dict.keys())].rename(columns=vars_dict).dropna()
    correlations = corr_df.corr()['LDL'].drop('LDL').sort_values(key=abs, ascending=True)
    colors = ["#de5568" if x > 0 else "#49cce3" for x in correlations.values]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index)
    ax.set_xlabel('Correlation Coefficient', fontweight='bold')
    ax.set_title('Nutrient & Poverty Correlations with LDL', fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('ds2500-project/Visualizations/correlation_bar_chart_LDL.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# PLOT 4: LINEAR REGRESSION SCATTER GRID
# ============================================================================

def plot_regression_scatter(df):
    female_df = df[df['RIAGENDR'] == 2]

    datasets = [
        (female_df, 'RIAGENDR', 'LBXTLG',  'Female Baseline vs Triglycerides'),
        (df,        'INDFMPIR', 'LBXTLG',  'Poverty Ratio vs Triglycerides'),
        (female_df, 'RIAGENDR', 'LBDLDLN', 'Female Baseline vs LDL'),
        (df,        'INDFMPIR', 'LBDLDLN', 'Poverty Ratio vs LDL'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (data, x_col, y_col, title) in zip(axes.flatten(), datasets):
        plot_df = data[[x_col, y_col]].dropna()
        X, y = plot_df[x_col].values, plot_df[y_col].values
        m, b = np.polyfit(X, y, 1)
        ax.scatter(X, y, alpha=0.2, s=5, color='#49cce3')
        ax.plot(np.linspace(X.min(), X.max(), 100), 
                m * np.linspace(X.min(), X.max(), 100) + b, color='#de5568', linewidth=2)
        ax.set_title(f'{title}\nR² = {r2_score(y, m * X + b):.3f}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    plt.suptitle('Linear Regression: Gender & Poverty vs Cholesterol', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ds2500-project/Visualizations/regression_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# REGRESSION SUMMARY
# ============================================================================

def regression_summary(df):
    features = ['RIDAGEYR', 'RIAGENDR', 'INDFMPIR', 'DR1TSFAT', 'DR1TFIBE', 'DR1TSUGR', 'DR1TTFAT']

    for target, name in [('LBXTLG', 'Triglycerides'), ('LBDLDLN', 'LDL')]:
        model_df = df[features + [target]].dropna()
        X, y = model_df[features], model_df[target]
        model = LinearRegression().fit(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = LinearRegression().fit(X_train, y_train).predict(X_test)

        print(f"\n=== {name} ===")
        print(f"  Train R²: {model.score(X, y):.3f}")
        print(f"  Test  R²: {r2_score(y_test, y_pred):.3f}")
        print(f"  RMSE:     {mean_squared_error(y_test, y_pred)**.5:.2f}")
        for feat, coef in sorted(zip(features, model.coef_), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feat}: {coef:.4f}")

# ============================================================================
# CORRELATION MATRIX
# ============================================================================

def plot_correlation_matrix(df):
    vars = {'RIDAGEYR': 'Age', 'RIAGENDR': 'Gender', 'INDFMPIR': 'Poverty',
            'DR1TSFAT': 'Sat Fat', 'DR1TFIBE': 'Fiber', 'DR1TSUGR': 'Sugar',
            'DR1TTFAT': 'Total Fat', 'LBXTLG': 'Triglycerides', 'LBDLDLN': 'LDL'}

    corr = df[list(vars.keys())].rename(columns=vars).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5, ax=ax)
    ax.set_title('Full Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig('ds2500-project/Visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    plot_nutrient_deficiency(df)
    plot_heatmaps(df)
    plot_correlation_bar(df)
    plot_regression_scatter(df)
    regression_summary(df)
    plot_correlation_matrix(df)

if __name__ == '__main__':
    main()