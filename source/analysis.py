import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm

from econml.dml import CausalForestDML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT= "/Users/yqchew/Documents/DataScienceInEconomics/EmpiricalProject/"
SOURCE = ROOT+"source/"
RESULTS = ROOT+"results/"

file1 = "emp_proj_data.csv"
emp_proj_data = pd.read_csv(SOURCE + file1)

#Crime Count descriptive statistics
total_crime_count = emp_proj_data['Crime Count'].sum()
print("Total Crime Count:", total_crime_count)
with open(RESULTS + 'crime_count_total.txt', 'w') as f:
    f.write(f"Total Crime Count:{total_crime_count}")

crime_count_desc = emp_proj_data['Crime Count'].describe()
with open(RESULTS + 'crime_count_descriptive_stats.txt', 'w') as f:
    f.write(str(crime_count_desc))

#Crime distribution across MSOAs
plt.figure(figsize=(10, 6))
plt.hist(emp_proj_data['Crime Count'], bins=100, color='red')
plt.xlabel('Crime Count')
plt.ylabel('Frequency')
plt.title('Distribution of Crime')
plt.savefig(RESULTS + 'Crime_distribution.png', dpi=600)

#Identify Top 10 MSOAs with highest crime rate
top_10_crime_rate = emp_proj_data.nlargest(10, 'Crime Rate per 1000 resident')
print("Top 10 Crime Rate:", top_10_crime_rate)

plt.figure(figsize=(10, 6))
plt.barh(top_10_crime_rate['MSOA21NM'], top_10_crime_rate['Crime Rate per 1000 resident'], 
        color='red')
plt.title('MSOAs in England and Wales with the Highest Crime Rates',
    fontweight='bold')
plt.xlabel('Crime Rate per 1000 resident')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(RESULTS + 'top_10_crime_rate_msoas.png', dpi=600)

#Regress Crime Rate on Unemployment Rate
print(emp_proj_data.isna().sum())

X= emp_proj_data['Unemployment Rate']
Y= emp_proj_data['Crime Rate per 1000 resident']
model1 = sm.OLS(Y, sm.add_constant(X))
result1 = model1.fit(cov_type='HC1')

with open(RESULTS + 'regression1.txt', 'w') as f:
    f.write(str(result1.summary()))

#Scatterplot with fitted regression line
predicted_values = result1.predict(sm.add_constant(X))

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='red',alpha=0.5)
plt.plot(X, predicted_values, color='blue', 
    label="Fitted Line")
plt.xlabel('Unemployment Rate')
plt.ylabel('Crime Rate per 1000 resident')
plt.xlim(0, 20) 
plt.ylim(0, 100)
plt.title('OLS Regression: Unemployment Rate vs Crime Rate',
    fontweight='bold')
plt.legend()
plt.savefig(RESULTS + 'regression1_plot.png', dpi=600)

#Regress Crime Rate on Unemployment Rate and Other Control Variables
w1 = emp_proj_data['Proportion_age_abv65']
w2 = emp_proj_data['Proportion_white']
w3 = emp_proj_data['Population Density: Persons per square kilometre']
w4 = emp_proj_data['Proportion_Level4_Edu']

X_with_controls = pd.DataFrame({'Unemployment Rate': X,
    'Proportion_age_abv65': w1,
    'Proportion_white': w2,
    'Population_density': w3,
    'Proportion_Level4_Edu': w4})

model2 = sm.OLS(Y, sm.add_constant(X_with_controls))
result2 = model2.fit(cov_type='HC1')

with open(RESULTS + 'regression2.txt', 'w') as f:
    f.write(str(result2.summary()))

#Add Interaction Terms
X_with_controls['Unemployment_PopDensity'] = \
    X_with_controls['Unemployment Rate'] * X_with_controls['Population_density']
X_with_controls['Unemployment_White']= \
    X_with_controls['Unemployment Rate'] * X_with_controls['Proportion_white']

model3 = sm.OLS(Y, sm.add_constant(X_with_controls))
result3 = model3.fit(cov_type='HC1')

with open(RESULTS + 'regression3.txt', 'w') as f:
    f.write(str(result3.summary()))



#Causal Forest
X = emp_proj_data[['Proportion_age_abv65', 'Proportion_white', 
    'Population Density: Persons per square kilometre', 'Proportion_Level4_Edu']]
T = emp_proj_data['Unemployment Rate']

cf = CausalForestDML(
    model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingRegressor(),
    random_state=123)

cf.fit(Y=Y, T=T, X=X, W=None)
ate = cf.ate(X)
ate_ci = cf.ate_interval(X)

with open(RESULTS + 'CausalForest_ATE.txt', 'w') as f:
    f.write(f"Estimated ATE:{ate} \nConfidence interval for ATE:{ate_ci}")

print("Estimated ATE:", ate)
print("Confidence interval for ATE:", ate_ci)

hte = cf.effect(X)

plt.figure(figsize=(10, 6))
plt.hist(hte, bins=50, color='lightblue', edgecolor='grey')
plt.axvline(x=ate, color='red', linestyle='--', label='ATE')
plt.xlabel('Treatment Effects')
plt.ylabel('Count')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend()
plt.savefig(RESULTS + 'HTE_histogram.png')
plt.clf()
