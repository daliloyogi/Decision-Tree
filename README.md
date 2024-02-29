https://www.kaggle.com/code/hossamahmedsalah/decision-tree-msp/

In this exercise I use the dataset on Kaggle "Drugs A, B, C, data analysis. These steps include data collection, model building, evaluation, and visualization of results.

Start by downloading, then importing the dataset and carrying out initial exploration to understand the structure and type of data in each column. This helps me figure out what kind of transformation is required for categorical data, which is then encoded using label encoding to convert it into a numeric format that can be processed by the model.

Next it is directed to divide the dataset into features and targets, and divide it into training and testing sets to prepare the model. In this exercise, I was directed to build a decision tree model using DecisionTreeClassifier from scikit-learn, and train it using training data.

After training the model, its performance is then evaluated using several metrics, including accuracy and more detailed classification reports. This evaluation gives me insight into how well my model is performing and in which areas it needs improvement.

The most interesting thing is the process of visualizing the tree decisions that have been made. Using plot_tree from scikit-learn, I can see the structure of the tree directly, understand which features have the most significant impact on decision making, and interpret how the model makes predictions.

This practice gave me a deep understanding of the concept and application of decision trees in data analysis. I also learned the importance of documenting proper data, careful model evaluation, and informative visualization in building effective models and understanding the results. With this foundation, I feel more confident in exploring and applying other machine learning techniques in the future.
