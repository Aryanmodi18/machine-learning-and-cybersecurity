import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
from nltk.util import pr
stemmer = nltk.snowball.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = stopwords.words('english')
pd.set_option('display.max_rows', None)
df = pd.read_csv('/Users/aryanmodi/Desktop/ml/hate_speech/DataSet.csv')
print(df.head())
df['label'] = df['hate _speech'].map({0:"no hate speech detected", 1:"hate speech detected"})

x = np.array(df['Words'])
y = np.array(df['label'])

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# set up email details
sender_email = "coddevs@harshmavani.email"
sender_password = "hXthJJj47vNW"
receiver_email = "prathambhoraniya09@gmail.com"
subject = "Hate speech detected"
message = "Hate speech has been detected in the text data you provided."

# get user input and check for hate speech
text_data = input("Enter text data: ")
df = cv.transform([text_data]).toarray()
prediction = clf.predict(df)

# send email if hate speech is detected
if prediction == 'hate speech detected':
    import smtplib
    with smtplib.SMTP("smtppro.zoho.in", port=587) as connection:
        connection.starttls()
        connection.login(user=sender_email, password=sender_password)
        connection.sendmail(
            from_addr=sender_email,
            to_addrs=receiver_email,
            msg=f"Subject: {subject}\n\n{message}"
        )
        print("Email notification sent.")
else:
    print("No hate speech detected.")