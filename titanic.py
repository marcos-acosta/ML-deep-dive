from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import re

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

''' PREPROCESSING '''

def process(df, verbose=False):
    # Ticket number will not be helpful to us and we don't have enough Cabin data
    df = df.drop(columns=['Ticket', 'Cabin'])
    df['Title'] = df['Name'].apply(lambda x: re.search(r', .+?\.', x).group()[2:-1])
    df['Title'] = df['Title'].apply(lambda x: x if (x in ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev']) else 'OTHER')

    # We are missing some Age and Embarked values so we will replace those with the most common values (not the best)
    df['Age'] = df['Age'].fillna(value=df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(value=df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(value=df['Fare'].mode()[0])

    # Scale SibSp, Age, Parch, and Fare to lie between 0 and 1
    df['SibSp'] = df['SibSp'].apply(lambda x: x/max(df['SibSp']))
    df['Age'] = df['Age'].apply(lambda x: x/max(df['Age']))
    df['Parch'] = df['Parch'].apply(lambda x: x/max(df['Parch']))
    df['Fare'] = df['Fare'].apply(lambda x: x/max(df['Fare']))

    # Now we will make Sex and Embarked one-hot to feed into the model
    sexes = pd.get_dummies(data=df['Sex'], prefix='Sex')
    embarkeds = pd.get_dummies(data=df['Embarked'], prefix='Embarked')
    classes = pd.get_dummies(data=df['Pclass'], prefix='Class')
    titles = pd.get_dummies(data=df['Title'], prefix='Title')
    df = pd.concat([df, sexes, embarkeds, classes, titles], axis=1)
    df = df.drop(columns=['Sex', 'Embarked', 'Pclass', 'Title_OTHER', 'Name'])

    if verbose:
        print(df.describe(include='all'))

    return df

train = process(train)
train, valid = train_test_split(train, test_size=0.2)
test = process(test)

''' DEFINING MODEL '''

features = ['Class_1', 'Class_2', 'Class_3', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title_Miss', 'Title_Mrs', 'Title_Master', 'Title_Dr', 'Title_Rev']
input_dim = len(features)
X_train = train[features]
X_valid = train[features]
X_test = test[features]
y_train = train['Survived']
y_valid = train['Survived']

def define_sequential_model(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def plot_history(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Validation loss'], loc='upper left')
    plt.show()

model = define_sequential_model(input_dim=input_dim)
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=300, batch_size=32)
plot_history(history.history)

# model = RandomForestClassifier(max_depth=5, n_estimators=200)
# model.fit(X_train, y_train)

''' PREDICT WITH MODEL '''

predictions = model.predict(X_test)
# Round predictions to 0 or 1
predictions = [int(round(x[0])) for x in predictions]

# predictions = model.predict(X_test)
# print(predictions)

results = pd.DataFrame()
results['PassengerId'] = test['PassengerId']
results['Survived'] = predictions

results.to_csv('out/titanic_results_nn_2.csv', index=False)