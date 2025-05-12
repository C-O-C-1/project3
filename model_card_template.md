# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- The model used is a Random Forest Classifiert.
- The model was trained by Cristina Ortiz Cruz as a part of the Udacity Course "Machine learning DevOps Engineer"
- The model was built using the Scikit-Learn library: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
- The model was trained using the default parameters provided by Scikit-Learn.

## Intended Use

- The model is intented to be used for income binary classification (income above or below $50K/yr) based on adult census data.
- The model is intended to be used for Social Science applications.

## Training Data

- For training the data, the "Census Income" data set was used (https://archive.ics.uci.edu/dataset/20/census+income) 
- The data was splited so that 80% of the data was used to train the model.
- The data set has 14 features:
    - Age, fnlwgt, education_num, capital-gain, capital-loss, hours-per-week --> categorical features
    - Workclass, education, marital-status, occupation, relationship, race, sex, native-country --> numerical features
- The income is the target variable that has to be predicted by the model.

## Evaluation Data

- For evaluating the the data, the 20% of the data set not using for training was used.
- The features described above apply to the evaluation data.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The metrics used for evaluating the model are:
- Precision
- Recall
- F-beta score (weighted harmonic mean of precision and recall)

The model performances on the test data are:

fbeta: 0.69
precision: 0.74
recall 0.65

We also computed the model performances on data slices taking into account categorical features:
For this extended analysis, refer to the provided file "slice_output.txt"

## Ethical Considerations
- The data does not contain information about individuals so that those could be identified.

## Caveats and Recommendations
- Missing values can affect/limit model performance. 
- Another interesting variable that could be studied is the amount of work experience or seniority level.
- The gender variable is binary: it could be extended to cover more genders.