#NOTES FOR LATER:
#Replace nutrient deficiency rates by poverty bar chart nutrients w/ clinically relevant & correlated variables
#TODO: focus on race & nutrition/obesity, explore blood pressure?, use K-Means clustering, linear regression

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

sns.set_theme()


# LOAD & MERGE

demographics_df = pd.read_sas('ds2500-project/NHANES_data/DEMO_L.xpt')
dietary_df = pd.read_sas('ds2500-project/NHANES_data/DR1TOT_L.xpt')
cholesterol_df  = pd.read_sas('ds2500-project/NHANES_data/TRIGLY_L.xpt')
blood_pressure_df = pd.read_sas('ds2500-project/NHANES_data/BPXO_L.xpt')
body_measures_df = pd.read_sas('ds2500-project/NHANES_data/BMX_L.xpt')

df = demographics_df.merge(dietary_df, on = 'SEQN', how = 'left').merge(cholesterol_df, on = 'SEQN', how = 'left').merge(body_measures_df, on = 'SEQN', how = 'left').merge(blood_pressure_df, on = 'SEQN', how = 'left')
df = df[df['DR1DRSTZ'] == 1] # only reliable dietary recall days have value = 1; drop all else

# add appropriate names to categorical variables
race_mapping = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White',
                4: 'Non-Hispanic Black', 6: 'Non-Hispanic Asian', 7: 'Other/Multi-racial'}
df['Race'] = df['RIDRETH3'].map(race_mapping)

df['Gender'] = df['RIAGENDR'].map({1: 'Male', 2: 'Female'})






# CLEANING, PREPROCESSING

for col in ['INDFMPIR', 'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR',
            'DR1TFIBE', 'DR1TTFAT', 'DR1TSFAT', 'DR1TCHOL', 'LBDLDLN', 'LBXTLG']:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

df = df.dropna(subset = ['RIDAGEYR', 'RIAGENDR', 'RIDRETH3'])

df['Poverty_Category'] = pd.cut(df['INDFMPIR'], 
                                bins = [0, 1, 2, 3, 5],
                                labels = ['<1.0 (Low)', '1.0-2.0 (Near Low)', '2.0-3.0 (Middle)', '>3.0 (High)'])

# capping extreme values at the 99th percentile to handle outliers
for col in ['LBXTLG', 'LBDLDLN', 'DR1TSFAT', 'DR1TSUGR', 'DR1TCHOL']:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper = cap)

# filtering to adults only for more relevance to nutritional differences
df = df[df['RIDAGEYR'] >= 18]

# assigning binary variables to obesity classification
df['BMI'] = df['BMXBMI']
df['Obese'] = (df['BMXBMI'] >= 30).astype(int)  # binary: 1 = obese, 0 = not obese





# CURRENT VISUALIZATIONS: OBESITY & RACE FOCUSED VISUALS

def plot_obesity_rate_by_race(df):
    race_order = ['Non-Hispanic Asian', 'Other Hispanic', 'Non-Hispanic White',
                  'Other/Multi-racial', 'Mexican American', 'Non-Hispanic Black']

    obesity_by_race = df.groupby('Race')['Obese'].mean() * 100
    obesity_by_race = obesity_by_race.reindex(race_order)

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.barh(race_order, obesity_by_race.values, alpha = 0.8, edgecolor = 'black', linewidth = 0.5)
    ax.axvline(x = obesity_by_race.mean(), color = 'black', linestyle = '--', label = f'Average ({obesity_by_race.mean():.1f}%)')
    ax.set_xlabel('Obesity Rate (%)', fontweight = 'bold')
    ax.set_title('Obesity Rate by Race', fontweight = 'bold')
    ax.legend(fontsize = 9)
    ax.grid(axis = 'x', alpha = 0.3)
    plt.tight_layout()
    plt.savefig('ds2500-project/Visualizations/obesity_rate_by_race.png', dpi = 300, bbox_inches = 'tight')
    plt.show()


def plot_weight_category_by_race(df):
    race_order = ['Non-Hispanic Asian', 'Other Hispanic', 'Non-Hispanic White',
                  'Other/Multi-racial', 'Mexican American', 'Non-Hispanic Black']

    df['Weight_Category'] = pd.cut(df['BMXBMI'], bins = [0, 18.5, 25, 30, np.inf],
                                   labels = ['Underweight (BMI <18.5)', 'Normal(BMI <25)', 'Overweight (BMI <30)', 'Obese (BMI >30)'])

    weight_pct = df.groupby('Race', observed = True)['Weight_Category'].value_counts(normalize = True).mul(100).unstack()
    weight_pct = weight_pct.reindex(race_order)

    fig, ax = plt.subplots(figsize = (10, 6))
    weight_pct.plot(kind = 'barh', stacked = True, ax = ax,
                    color = ["#b8e4ff", "#7080fa", "#563cc6", "#27003F"],
                    alpha = 0.8, edgecolor = 'black', linewidth = 0.3)
    ax.set_xlabel('Percentage (%)', fontweight = 'bold')
    ax.set_title('Weight Category Distribution by Race', fontweight = 'bold')
    ax.legend(title = 'Weight Category', bbox_to_anchor = (1.05, 1), loc = 'lower right', fontsize = 9)
    ax.grid(axis = 'x', alpha = 0.3)
    plt.tight_layout()
    plt.savefig('ds2500-project/Visualizations/weight_category_by_race.png', dpi = 300, bbox_inches = 'tight')
    plt.show()


def plot_nutrient_intake_by_race(df):
    race_order = ['Non-Hispanic Asian', 'Other Hispanic', 'Non-Hispanic White',
                  'Other/Multi-racial', 'Mexican American', 'Non-Hispanic Black']

    nutrients  = ['DR1TKCAL', 'DR1TFIBE', 'DR1TSUGR', 'DR1TSFAT']
    titles     = ['Average Calorie Intake', 'Average Fiber Intake', 'Average Sugar Intake', 'Average Saturated Fat Intake']
    xlabels    = ['Calories (kcal)', 'Fiber (g)', 'Sugar (g)', 'Saturated Fat (g)']
    guidelines = [2000, 25, 50, 20]
    colors     = ["#b8e4ff", "#7080fa", "#563cc6", "#27003F"]

    fig, axes = plt.subplots(2, 2, figsize = (18, 12))

    for ax, nutrient, title, xlabel, guideline, color in zip(axes.flatten(), nutrients, titles, xlabels, guidelines, colors):
        means = df.groupby('Race')[nutrient].mean().reindex(race_order)
        ax.barh(race_order, means.values, color = color, alpha = 0.8, linewidth = 0.5)
        ax.axvline(x = guideline, color = 'black', linestyle = '--', label = f'Guideline ({guideline})')
        ax.set_xlabel(xlabel, fontweight = 'bold')
        ax.set_title(title, fontweight = 'bold')
        ax.legend(fontsize = 9)
        ax.grid(axis = 'x', alpha = 0.3)

    plt.suptitle('Nutrient Intake by Race', fontsize = 14, fontweight = 'bold')
    plt.tight_layout()
    plt.savefig('ds2500-project/Visualizations/nutrient_intake_by_race.png', dpi = 300, bbox_inches = 'tight')
    plt.show()


def plot_bmi_by_calorie_quartile_and_race(df):
    df['Calorie_Quartile'] = pd.qcut(df['DR1TKCAL'], q = 4, 
                                      labels = ['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    pivot = df.groupby(['Race', 'Calorie_Quartile'], observed = True)['BMXBMI'].mean().unstack()
    
    fig, ax = plt.subplots(figsize = (12, 6))
    pivot.plot(kind = 'bar', ax = ax, colormap = 'coolwarm', alpha = 0.8, edgecolor = 'black', linewidth = 0.5)
    ax.axhline(y = 30, color = 'black', linestyle = '--', label = 'Obesity threshold')
    ax.set_xlabel('Race', fontweight = 'bold')
    ax.set_ylabel('Mean BMI', fontweight = 'bold')
    ax.set_title('Mean BMI by Race and Calorie Intake Quartile', fontweight = 'bold')
    ax.legend(title = 'Calorie Quartile')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha = 'right')
    plt.tight_layout()
    plt.show()

def main():
    plot_obesity_rate_by_race(df)
    plot_weight_category_by_race(df)
    plot_nutrient_intake_by_race(df)
    # plot_bmi_by_calorie_quartile_and_race(df)

if __name__ == '__main__':
    main()