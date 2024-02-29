import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# mengimpor kelas tempat DT diimplementasikan 
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR 

# import drugs dataset
drugs = pd.read_csv('/content/archive.zip')
drugs

# mendapatkan pairplot dari data 'drugs' dengan setiap titik berwarna sesuai dengan kategori yang diberikan oleh kolom 'Drug'
sns.pairplot(data=drugs, hue='Drug');

# memberikan informasi tentang struktur DataFrame drugs, termasuk informasi tentang jumlah baris dan kolom, jenis data di setiap kolom, dan jumlah nilai non-null di setiap kolom.
drugs.info()

# let's see the categorical features
drugs.describe(include='O')

# We have some categorical features let's encode them 
# we can use one_hot_encodeing 
# or we can use labee_encoding 
# let's chooce label_encoding for this problem 
from sklearn.preprocessing import LabelEncoder

# initiating the class
label_enc = LabelEncoder()

# columns that are categorical 
cols = drugs.select_dtypes(include='O').columns
# looping on each column in the dataset
for col in cols:
    # Label encoding each column 
    drugs[col] = label_enc.fit_transform(drugs[col])

# displaying the data after encoding 
drugs

# dividing the data into X, y 
# X: the features 
# y : the target 🎯 
X = drugs.drop(columns='Drug')
y = drugs['Drug']

# let's see our data X, y
display(X.head(3), y.head(3))

# now we need to split the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)

# Building the model 
tree_clf0 = DTC()
# Fitting the model
tree_clf0.fit(X_train,y_train)

# let's see the model score (acc) on the training set 
tree_clf0.score(X_train, y_train)

# another way of calculating the accuracy
from sklearn.metrics import accuracy_score, classification_report
y_pred = tree_clf0.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy

# third way using more than one metric 
report = classification_report(y_test, y_pred)
print(report)

# visualsing👀 the Decision Tree🌳 
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 8), dpi=200)
plot_tree(tree_clf0, feature_names=drugs.columns, filled=True);
# Optinal parameters
# feature_names=drugs.columns, filled=True  
# filled=True  colors
