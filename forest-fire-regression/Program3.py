'''
CIS-443-01
program / assignment 3
due 4/4/25

this program runs a linear regression on a provided forest fire dataset to determine how different
weather factors change the size of a forest fire. it makes different models for each independent var
and runs the regression to get the needed stats, makes predictions and then visualizes the models

'''


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

df = pd.read_csv("forestfires.csv")

# define variables and prediction inputs
models_info = [
    ("Temperature", 23),
    ("Humidity", 95),
    ("Wind", 8),
    ("Rain", 1),
]

model_results = []

for feature, pred_value in models_info:
    x = df[feature]
    y = df["Fire_Area"]

    # run linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_pred = intercept + slope * x
    prediction = intercept + slope * pred_value

    # plot
    plt.figure()
    plt.scatter(x, y, alpha=0.5, label='Data')
    plt.plot(x, y_pred, color='red', label='regression line')
    plt.title(f'{feature} vs Fire_Area')
    plt.xlabel(feature)
    plt.ylabel('Fire_Area')
    plt.grid(True)
    plt.legend()
    plt.show()

    # add results
    model_results.append({
        "feature": feature,
        "slope": slope,
        "intercept": intercept,
        "r-squared": r_value**2,
        "p-value": p_value,
        "standard error": std_err,
        "predicted Fire_Area": prediction
    })

# print results
print("\nlinear regression results:\n")
for result in model_results:
    print(f"--- {result['feature']} ---")
    print(f"slope: {result['slope']:.6f}")
    print(f"intercept: {result['intercept']:.6f}")
    print(f"r-squared: {result['r-squared']:.6f}")
    print(f"p-value: {result['p-value']:.6f}")
    print(f"standard error: {result['standard error']:.6f}")
    print(f"prediction (Fire_Area at {result['feature']}={models_info[[r['feature'] for r in model_results].index(result['feature'])][1]}): {result['predicted Fire_Area']:.6f}\n")


'''
see comments on BB for conclusion / explanation (I could not find a good way to format 
it in comments here and my understanding is that it does not need to be a printed result)
'''