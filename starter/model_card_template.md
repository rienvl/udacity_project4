# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This purpose of the model is to estimate if the income of an individual is below or above 50k.
The model is a standard logistic regression model from sklearn ("LogisticRegression"). 
Parameter C (: the inverse of regularization strength) was set to 1.0.
The model was trained on the UCI Census Income Dataset; a dataset meant for research purposes.

## Intended Use
The dataset is purely for research purposes and of poor quality.
The effort put in model training is therefore limited and no conclusions should be drawn from its output.  

## Training Data
Input data contains the following features:
6 continuous  features: {age, capital-gain, capital-loss, education-num, fnlgt, hours-per-week}
8 categorical features: {workclass, education, marital-status, occupation, relationship, race, sex, native-country}
And class label "salary".
Using a 80/20% train/test split resulted in 24129 valid training samples.

## Evaluation Data
After test-split, 6033 data samples remained for evaluation.

## Metrics
The model computes the f1-score, precision, and recall.
The model performance for each metrics after training for 100 epochs is:
f1_score = 0.3690,   precision = 0.6963,   recall = 0.2510

## Ethical Considerations
The set of feature we use is very small relative to all features that influence income in real life.
Furthermore, the dataset suffers from class-imbalance (e.g. for input features "Race" and "Sex"),
as well as for the output feature 'salary'. 
  
## Caveats and Recommendations
As mentioned earlier, the dataset used for training is of poor quality.
First recommendation is to collect more data for the purpose of reducing the class imbalances.

