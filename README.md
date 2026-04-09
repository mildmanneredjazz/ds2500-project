# NHANES Racial Demographic & Obesity Analysis

An exploratory data analysis of the CDC's National Health and Nutrition Examination Survey (NHANES) 2021–2023 dataset, investigating relationships between race, dietary intake, and supplemental exploration into gender and nutritional disparities.

---

## Data Sources

All data comes from the [CDC NHANES 2021–2023 cycle](https://wwwn.cdc.gov/nchs/nhanes/). Files can be downloaded directly from the NHANES website:

- **DEMO_L** — Demographics
- **DR1TOT_L** — Dietary Interview (Total Nutrient Intakes)
- **BMX_L** — Body Measures

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
```

Install all dependencies with:

```
pip install pandas numpy matplotlib seaborn
```

---

## Preprocessing Steps

- Merges all five datasets on `SEQN` (unique participant ID)
- Filters to reliable dietary recall days only (`DR1DRSTZ == 1`)
- Removes negative values from key dietary variables/family income to poverty ratio
- Filters to adults only (`RIDAGEYR >= 18`)
- Caps outliers at the 99th percentile for dietary variables
- Creates income categories from the poverty-to-income ratio (`INDFMPIR`)
- Adds binary obesity classification of BMI >30 = "Obese", BMI <30 = Not (`BMXBMI >= 30`)

---

## Visualizations

| Function | Description |
|---|---|
| `plot_obesity_rate_by_race` | Horizontal bar chart of obesity rates by racial group |
| `plot_weight_category_by_race` | Stacked bar chart of BMI weight categories by race |
| `plot_nutrient_intake_by_race` | 2x2 grid comparing calorie, fiber, sugar, and saturated fat intake by race |
| `box_plot_bmi_race` | Box plot of BMI distribution by race with obesity threshold |
| `plot_obesity_rate_by_race_and_gender` | Side-by-side obesity rates by race, split by gender |

---

## Key Findings

- **Race is a strong axis of varation in obesity outcomes/trends in average BMI** — Non-Hispanic Asian participants show consistently lower BMI, obesity rates (~16%), and excess nutrient intake across every visualization, while Non-Hispanic Black (~49%) and Mexican American (~47%) participants show the highest obesity rates, both well above the sample average of 39.4%.
- **There are significant gender gaps within racial groups as it relates to BMI** — Non-Hispanic Black females show the single highest obesity rate of any group (~55%), compared to ~42% for Non-Hispanic Black males. In comparison, Mexican American males and females show little gender difference (~46-47%), while Non-Hispanic Asian females have the lowest obesity rate of any group (~14%).
- **Certain nutrient intake differences align with patterns of obesity** — Non-Hispanic Asian participants consistently show the lowest sugar and saturated fat intake, which may play a role in lower obesity rates. Still, it is important to highlight that there are also divergences in expected nutrient intake patterns as they relate to BMI, with Other/Multi-racial and Non-Hispanic White individuals showing the highest average excess consumption of calories, sugar, and saturated fat intake, yet do not represent groups with the highest average BMI.

---

## Usage

Place all `.xpt` data files in `ds2500-project/NHANES_data/`, then run:

```
python NHANES_BloodLipid_Analysis.py
```

All visualizations will be saved to `ds2500-project/Visualizations/`.
