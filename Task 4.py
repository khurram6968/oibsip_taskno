import pandas as pd
import os
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset from CSV
file_path = "E:\datascience"
os.chdir(file_path)
df = pd.read_csv("spam.csv",encoding='ISO-8859-1')

df = df.dropna(axis=1, how='any')
df = df.rename(columns={'v1': 'Category', 'v2': 'Message'})

import nltk
nltk.download('stopwords')

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)


# Apply preprocessing to each message
df['cleaned_text'] = df['Message'].apply(preprocess_text)

# Assuming the 'Category' column contains the labels 'spam' and 'ham'
df['label'] = df['Category'].replace({'spam': 1, 'ham': 0})

# Verify the conversion
print(df['Category'].value_counts())

# Convert Text to Numerical Data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']  # Assuming 'label' column has spam (1) or ham (0) values


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Test the Model on New Emails
def predict_spam_or_not_spam(text):
    processed_text = preprocess_text(text)
    transformed_text = vectorizer.transform([processed_text])
    prediction = model.predict(transformed_text)
    return 'Spam' if prediction == 1 else 'Not Spam'

# Test the function
email_text = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim."
print(predict_spam_or_not_spam(email_text))

# Test the function
email_text = "Hey John, I just wanted to check in and see if you’re free this weekend for a quick catch-up. Let me know when works for you!"
print(predict_spam_or_not_spam(email_text))