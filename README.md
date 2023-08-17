# String
- A Machine Learning Framework
- Designed Using: Python and NumPy

# Linear Regression:
A statistical process for estimating the relationships between a dependent variable (`outcome/response`) and one or more independent variables (`features`, `covariates`, `predictors`, or `explanatory variables).
 
 ## Simple Regression
 Estimating the relationships between two quantitative variables.
 * Independent variable/ outcome/ predicted value: Body weight (pounds/kg)
 * Input Feature/predictors/ input value: height (inches/cm)
   ![SimpleLinearRegression](https://github.com/Supertring/ml-framework/assets/81057028/a3072c93-ef59-4a4e-906e-43e149bcc2c2)

 ## Multiple variable Regression
 Determine the relationship between several independent variables and a dependent variable.
 * Independent variable/ outcome/ predicted value: Sales
 * Input Feature/predictors/ input value: the price of the product, interest rates, and competitive price

 ## Applications of Linear Regression
 ### Economics and Finance:
 * __Stock Market Analysis__:Predict stock prices or returns based on historical data and other relevant financial indicators.
 * __Econometric modeling__: Analyzing the relationship between economic variables like GDP, inflation, and unemployment.

 ### Marketing and Business:
 * __Sales Forecasting__: Predicting future sales based on factors like advertising spending, product pricing, and market trends.
 * __Customer Behavior Analysis__: Understand how customer behavior (eg:., website visits, clicks) relates to sales or other outcomes.

 ### Healthcare:
* __Drug Dosage Prediction__: Determine appropriate drug dosages based on patient characteristics.
* 
 ### Social Sciences:
 * __Psychology Research__: Analyzing the relationship between variables like time spent studying and exam scores.
 * __Sociology Studies__: Relation between Demographic factors and social behavior.

 ### Environmental Sciences:
 * __Climate Modeling__: Predicting temperature changes or sea levels based on historical climate data and relevant variables.

 ### Engineering:
 * __Quality Control__: Relationships between production parameters and product quality.
 * __Process Optimization__: Optimizing manufacturing processes by analyzing the impact of different factors on output quality.

 ### Sports Analytics:
 * __Player Performance Prediction__: Predicting a player's performance based on historical statistics and game conditions.

 ### Education:
 * __Student Performance Analysis__: Predict scores based on factors like study time, attendance, and socioeconomic background.

 ### Real Estate:
 * __Property Price Prediction__: Predict property based on features like location, size, and local economic indicators.

 ### Energy Consumption Analysis:
 * __Energy Demand Forecasting__: Predicting energy demand based on historical consumption patterns and weather conditions.

# Data Preparation for Linear Regression
The quality of data and preprocessing directly impact the accuracy and interpretability of your results. Here are the key steps in data preparation for linear regression:

## Data Collection and Inspection
* __Gather data__: dependent variable (target) and independent variable (predictors).
* __Inspect__: the dataset for its structure, size, and variable types.
* __Identify__: the dependent (y) and independent variables (x1,x2,x3,.....,xn).
* __Check__: for missing values, outliers, and anomalies in the data.

## Data Cleaning
* __Handle missing values__: Decide whether to remove or impute missing values based on the nature of the data.
* __Imputation techniques__: Mean, median, mode imputation, or using predictive models to impute missing values.
* __Outlier Detection__: Identify outliers that might negatively affect the prediction model. Visualize using (box plots, scatter plots) and statistical methods (z-score, IQR)
* __Decide__: whether to remove, transform, or treat outliers based on domain knowledge.
* __Consider__: techniques like winsorization, log transformation, or replacing outliers with a reasonable value.

## Feature Selection
* __Identify relevant features__: Examine the datasets to select the most relevant predictor variables (feature input).
* __Exclude__: variables that are irrelevant or might introduce multicollinearity.
* __Multicollinearity__: statistical concept where several independent variables in a model are correlated.

## Feature Transformation
* __Categorical variables__: Convert categorical variables into numerical representations
  * __One-hot encoding__: for nominal variables (information to distinguish objects: eg: zip code, employee id, eye color, gender: 🕵️‍♂️, female)
  * __Label encoding__: for ordinal variables (enough information to order objects: hardness of minerals, grades, street numbers, quality:{good, better, best})
* __Scaling__: Normalize or standardize numerical features to ensure they are on the same scale.
  * This helps prevent variables with larger magnitudes from dominating the model.
  * Mix-max scaling
  * Standardization (z-score normalization)

## Data Splitting
* __Divide dataset__: into training and testing substes. The training set is used to train the model, and the testing set is used to evaluate its performance.
   * Common split: 80-20 or 70-30 for training and testing respectively. 

## Feature Engineering
* __Create New Features__: Generate new features by combining existing ones,
   * Or applying a mathematical transformation (e.g.: squaring, logarithm)
   * Or creating interaction terms (a multiplication of two features that you believe have a joint effect on the target)
   * Or polynomial features to capture more complex relationships
   * For example, if an input sample is two-dimensional and of the form [a, b], the degree-2 polynomial features are $[1, a, b, a^2, ab, b^2]$.

## Multicollinearity Handling
* Calculate the correlation matrix among feature variables to identify highly correlated pairs.
* Consider correlation threshold (eg. 0.8/0.7) to identify multicollinearity.
* Remove one of the correlated variables if they provide similar information.
* Check for multicollinearity (high correlation) among predictor(features) variables, as it can lead to unstable coefficient estimates.
* Use dimensionality reduction techniques (PCA) if multicollinearity is severe.

## Residual Analysis
* Fit a preliminary linear regression model using training data
* Analyze the residuals (differences between predicted and actual values)
* Check for patterns, unequal spread of residuals, and outliers in residual plots.

## Feature Testing and Transformation
* If necessary transform the feature variables to achieve linearity.
* Use scatter plots and partial regression plots to assess linearity.
* Techniques like logarithmic, or exponential transformations can help.

## Model Building and Evaluation
* Train the linear regression model using the training data
* Evaluate the model's performance on testing data: use MSE, RMSE, etc.
* Interpret the model coefficients to understand the relationship between feature variables and the expected output variable.

## Model Improvement
* Based on the model evaluation, iteratively refine the model.
* By adjusting feature selection, address issues identified in the residual analysis
* Try different transformations.

__Linear regression assumes a linear relationship between variables, if your results do not show linear patterns, you might consider other regression techniques or non-linear models. Preparation of data is an iterative process that requires careful consideration, domain knowledge, and experimentation to build a robust linear regression model.__


Algorithms Implemented
------------------------------------------
- Naive Bayes
- Linear Regression
- Logistic Regression
- KMeans
- Decision Tree
- Perceptron
- Support Vector Machines
