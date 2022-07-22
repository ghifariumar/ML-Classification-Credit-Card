# ML-Classification-Credit-Card

- Create Machine Learning Classification Model to detect Credit Card fraud
- Minimize false negative error Recall(+) => model predict does not fraud while it does fraud
- Trying over sampling, under sampling, SMOTE, and class weighting
- Dataset Source : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
```
- Read Dataset
<img width="587" alt="image" src="https://user-images.githubusercontent.com/99155979/180383316-c69664a6-1d54-4330-85f1-55acd9d0ee54.png">

- Check Missing Value
```python
plt.figure(figsize=(13,8))
sns.heatmap(df.isna(),cmap = 'viridis', cbar = False, yticklabels=False)
plt.show()
```
<img width="500" alt="image" src="https://user-images.githubusercontent.com/99155979/180384535-309688f3-ec37-4390-9352-dd349a2f745e.png">


- Check Imbalance Dataset
```python
pd.crosstab(index=df['Class'], columns='count', normalize = True)*100
```
| **Class** | **Count** |
| --- | --- |
| 0 | 99.827 |
| 1 | 0.172 |

- Create Model to Detect Fraud
  - 1 => Fraud
  - 2 => Not-Fraud

- Splitting Data
<img width="553" alt="image" src="https://user-images.githubusercontent.com/99155979/180384815-c0a8bd31-2960-431b-904d-9342a9c91815.png">

- Machine Learning Modelling
<img width="215" alt="image" src="https://user-images.githubusercontent.com/99155979/180390240-560b95f8-5bfb-4915-868e-fc5baf12c0dd.png">
  - Train
  <img width="294" alt="image" src="https://user-images.githubusercontent.com/99155979/180390315-0608406f-78fc-4fe2-91ed-d36c0f96b318.png">
  - Test
  <img width="307" alt="image" src="https://user-images.githubusercontent.com/99155979/180390402-7026c3c7-c4b6-4eab-ba3e-f27daf2b1ddc.png">

```python
sns.heatmap(df_LR_ts, annot=True, cbar=False)
plt.show()
```
<img width="296" alt="image" src="https://user-images.githubusercontent.com/99155979/180390616-d79052ef-b7c1-40da-9219-165cf56435da.png">

## Random Over Sampling
  - **Duplicating data randomly** class-target minority (class 1) until it has the same amount with class-target majority (class 0).
  - **Fraud** dataframe will be over sampling until it has the same amount with **Non Fraud** dataframe.

```python
fraud_oversample = resample(fraud, replace=True, n_samples=len(non_fraud), random_state = 42)
df_Oversample = pd.concat([non_fraud,fraud_oversample])
df_Oversample['Class'].value_counts()
```
<img width="154" alt="image" src="https://user-images.githubusercontent.com/99155979/180391503-1edbe997-74be-465e-bea7-60cb97f8a1c4.png">

============================================================================================

<img width="332" alt="image" src="https://user-images.githubusercontent.com/99155979/180391664-9d418fdc-3513-4df0-bd42-3442e7c75a1c.png">

```python
sns.heatmap(df_OS, annot=True, cbar=False)
plt.show()
```
<img width="302" alt="image" src="https://user-images.githubusercontent.com/99155979/180392011-1e7c8f48-b78f-4a2e-ba31-a8dfb9fa6757.png">

============================================================================================

## Random Under Sampling
  - **Remove data randomly** in majority class(clas 0) until it has the same amount with (class 1)
  - **Non Fraud** data frame will be undersampling until it has the same amount with **Fraud** dataframe
  - Under Sampling rarely to be used. It has a chance to lose some information

```python
non_fraud = df_train[df_train['Class']==0] ## majority class
fraud = df_train[df_train['Class']==1] ## minority class

non_fraud_Undersample = resample(non_fraud, # majority class
                                 replace=False,
                                 n_samples=len(fraud), #minority class
                                 random_state = 42)
                                 
df_Undersample = pd.concat([non_fraud_Undersample, fraud])

df_Undersample['Class'].value_counts()
```
<img width="146" alt="image" src="https://user-images.githubusercontent.com/99155979/180392571-5a208745-e161-4fe9-870b-dc32fa6615bc.png">

============================================================================================

<img width="328" alt="image" src="https://user-images.githubusercontent.com/99155979/180392903-6cf2e0ad-eb6f-47b5-a0c7-eb7bf586faf9.png">

```python
sns.heatmap(df_US, annot=True, cbar=False)
plt.show()
```
<img width="290" alt="image" src="https://user-images.githubusercontent.com/99155979/180392982-5f4a654a-af03-413f-95d8-67df487e8450.png">

============================================================================================
## SMOTE - Synthetic Minority Oversampling Technique
- **Create Synthetic Randomly** from minoruty class (Class 1), until it has the same amount with majority class (Class 0)

- Import and install package
```python
import imblearn
conda install -c conda-forge imbalanced-learn==0.6
from imblearn.over_sampling import SMOTE
```
```python
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

# Optional for checking proportion
df_SMOTE = pd.concat([X_train_sm, y_train_sm], axis=1)
df_SMOTE['Class'].value_counts()
```
<img width="160" alt="image" src="https://user-images.githubusercontent.com/99155979/180421319-e0a020db-ae0c-4d7c-82f8-c210d52bc3c6.png">

```python
LR_SMOTE = LogisticRegression()
LR_SMOTE.fit(X_train_sm, y_train_sm)
y_predSMOTE = LR_SMOTE.predict(X_test)
print(classification_report(y_test, y_predSMOTE))
```
<img width="300" alt="image" src="https://user-images.githubusercontent.com/99155979/180421465-70978daf-8df7-43f9-b6fd-270ad602cf57.png">

```python
cm_SMOTE = confusion_matrix(y_test, y_predSMOTE, labels=[1,0])
df_SMOTE = pd.DataFrame(cm_SMOTE, index=['Akt 1', 'Akt 0'], columns=['Pred 1', 'Pred 0'])
sns.heatmap(df_SMOTE, annot=True, cbar=False)
plt.show()
```
<img width="277" alt="image" src="https://user-images.githubusercontent.com/99155979/180421680-99fcf4b7-1362-4879-8f72-3e8561274e04.png">

## Class Weight
- With class weighting, no need to use SMOTE or random sampling
- Using the first X_train and  y_train
- Focusing on learning the model
- Total Class Weight = 1

```python
LR_CW = LogisticRegression(class_weight={0:0.3, 1:.97})
LR_CW.fit(X_train, y_train)
y_predCW = LR_CW.predict(X_test)
print(classification_report(y_test, y_predCW))
```
<img width="307" alt="image" src="https://user-images.githubusercontent.com/99155979/180426334-46a823c4-ab54-4f47-beb3-344c88b90935.png">

```python
cm_CW = confusion_matrix(y_test, y_predCW, labels=[1,0])
df_CW = pd.DataFrame(cm_CW, index=['Akt 1', 'Akt 0'], columns = ['Pred 1','Pred 0'])
sns.heatmap(df_CW, annot=True, cbar=False)
plt.show()
```
<img width="281" alt="image" src="https://user-images.githubusercontent.com/99155979/180426542-7e696c99-97c8-4385-9e55-88bbc171ca95.png">

