# Predicting Accident Severity in Maryland with Machine Learning

This project, developed for the course (Data Mining & Visualization), aims to enhance road safety in Maryland, USA, by predicting car crash severity using machine learning techniques. The predictive model assesses the extent of damage and passenger injury to inform policy recommendations for traffic regulation implementation.

## Project Overview
- Goal: Develop a model to predict car crash severity in Maryland, using it as a basis for data-driven policy recommendations to improve road safety.
- Data Sources: The primary dataset used is "Crash Reporting - Drivers Data" from Montgomery County, Maryland, chosen for its comprehensive metadata and relevance.
- Techniques Employed:
  - Data Pre-processing: Initial cleaning, handling missing values, categorical encoding, and addressing class imbalance.
  - Exploratory Data Analysis (EDA): Unveiling patterns, and relationships in crash data to inform modeling.
  - Predictive Modeling: Utilizing Random Forest, XGBoost, and Logistic Regression, focusing on interpretability, performance, and predictive capability.
  - Class Imbalance Mitigation: Employing undersampling to enhance model performance.
  - Hyperparameter Tuning: Optimizing model parameters for improved accuracy.
## Key Findings
- The model demonstrates the ability to predict high-emergency crashes with significant accuracy, highlighting the importance of specific variables like 'Vehicle Damage Extent' in injury prediction.
- SHAP analysis provides insights into the impact of individual features on the prediction outcome, aiding in understanding model decisions.
## Policy Recommendations
- Enhance emergency response procedures based on feature importance insights.
- Implement targeted safety measures for high-risk areas identified through geospatial analysis.
## Tools and Technologies
- Python for data analysis and modeling, including libraries like Pandas, Scikit-learn, and XGBoost.
- Streamlit for deploying an interactive web demo to showcase model predictions.
## Future Directions
- Exploring neural networks for potentially improved predictions, albeit with complexity and interpretability considerations.
- Investigating further into geographical clustering to optimize emergency response locations.
For more details on the methodology, model evaluation, and insights, please refer to the full report and code in this repository.
