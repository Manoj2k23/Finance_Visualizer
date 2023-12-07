#!/usr/bin/env python
# coding: utf-8

# In[59]:


# Import necessary libraries
import pandas as pd  # Import Pandas for data manipulation
import seaborn as sns  # Import Seaborn for data visualization
import matplotlib.pyplot as plt  # Import Matplotlib for custom plots
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
from sklearn.svm import SVC  # Import Support Vector Machine (SVM) classifier
from sklearn.metrics import confusion_matrix, classification_report  # Import metrics for evaluation
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Import preprocessing tools
from sklearn.model_selection import train_test_split  # Import train-test split function
from sklearn.metrics import accuracy_score  # Import accuracy metric

%matplotlib inline  # Display Matplotlib plots inline


# Define the file path to the CSV data file
file_path = 'D:/SRM CSE20-24/7th sem/Minor project/sklearn-expense-tracker-main/sklearn-expense-tracker-main/classified_data.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)


# In[61]:


df.head()


# In[62]:


df.info()


# In[63]:


# Check for missing values in the DataFrame
# The result will show the count of missing values for each column
missing_values_count = df.isnull().sum()


# In[64]:


# Retrieve unique values in the 'Category' column of the DataFrame
unique_categories = df['Category'].unique()



# In[65]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to the 'Category' column
# This assigns a unique numerical label to each distinct category
df['Category'] = label_encoder.fit_transform(df['Category'])



# In[67]:


df.head()


# In[68]:


# Count the occurrences of each unique value in the 'Category' column
value_counts = df['Category'].value_counts()



# In[69]:


# Create a count plot using Seaborn to visualize the distribution of categories
sns.countplot(df['Category'])



# In[70]:


# Split the DataFrame into feature data (X) and target data (y)
X = df.drop('Category', axis=1)  # X contains all columns except 'Category'
y = df['Category']  # y contains the 'Category' column as the target variable


# Split the feature and target data into training and testing sets
# X_train and y_train are for training, X_test and y_test are for testing
# test_size specifies the proportion of the data to be used for testing (20% in this case)
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


sc = StandardScaler()


# Convert the 'Transaction Date' column in training and testing sets to datetime objects
X_train['Transaction Date'] = pd.to_datetime(X_train['Transaction Date'])
X_test['Transaction Date'] = pd.to_datetime(X_test['Transaction Date'])


# Extract year, month, and day as separate columns
X_train['Year'] = X_train['Transaction Date'].dt.year
X_train['Month'] = X_train['Transaction Date'].dt.month
X_train['Day'] = X_train['Transaction Date'].dt.day

X_test['Year'] = X_test['Transaction Date'].dt.year
X_test['Month'] = X_test['Transaction Date'].dt.month
X_test['Day'] = X_test['Transaction Date'].dt.day

# Now, you have 'Year', 'Month', and 'Day' columns as numeric features.
# You can use them along with other numeric features for your Random Forest Classifier.


# In[74]:


print(X_train.columns)


# In[75]:


X_train['Transaction Date'] = pd.to_datetime(X_train['Transaction Date'])
X_test['Transaction Date'] = pd.to_datetime(X_test['Transaction Date'])


# In[76]:


# Prepare data
numeric_features = ['Amount', 'Year', 'Month', 'Day']
X_train = X_train[numeric_features]
X_test = X_test[numeric_features]

# Create and train RFC
rfc = RandomForestClassifier(n_estimators=200, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions
y_pred = rfc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest Classifier: {accuracy:.2f}")


# In[81]:


#Let's see how our model performed
print(classification_report(y_test, y_pred))


# In[82]:


# Assuming you have a DataFrame 'df' with non-numeric columns 'Province', 'City', 'Description'
# and a target column 'TargetColumn'

# Step 1: Load your data and preprocess non-numeric columns
label_encoder = LabelEncoder()

# Convert non-numeric columns to numeric using label encoding
df['Province'] = label_encoder.fit_transform(df['Province'])
df['City'] = label_encoder.fit_transform(df['City'])
df['Description'] = label_encoder.fit_transform(df['Description'])

# Select non-numeric columns
X = df[['Province', 'City', 'Description']]

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Description'], test_size=0.2, random_state=42)

# Step 3: Create and train the RFC model
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = rfc.predict(X_test)

# Step 5: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest Classifier: {accuracy:.2f}")


# In[ ]:




