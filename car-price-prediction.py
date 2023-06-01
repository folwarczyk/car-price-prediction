import pandas as pd

#loading the data
df1 = pd.read_csv('datasets/cars.csv')
df2 = pd.read_csv('datasets/autoscout24-germany-dataset.csv')

df1 = df1.drop(columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8','feature_9', 'feature_1', 'has_warranty', 'state', 'is_exchangeable', 'location_region', 'number_of_photos', 'up_counter', 'duration_listed', 'engine_type'])
df2 = df2.drop(columns='offerType')

df1 = df1.rename(columns ={
    'odometer_value':'mileage',
    'manufacturer_name': 'make',
    'model_name': 'model',
    'engine_fuel':'fuel',
    'transmission': 'gear',
    'price_usd': 'price',
    'year_produced': 'year'
})

#changing data types
df1['mileage'] = df1['mileage'].astype(float)
df1['price'] = df1['price'].astype(float)

df2['mileage'] = df2['mileage'].astype(float)
df2['price'] = df2['price'].astype(float)

df = pd.concat([df1, df2], ignore_index=True)

#data cleaning:
for column in ['year', 'mileage', 'engine_capacity', 'hp']:
    df[column] = df[column].fillna(df[column].median())

#data cleaning: filling missing values with mod (most common value
for column in ['model', 'gear', 'color', 'engine_has_gas', 'body_type', 'drivetrain']:
    df[column] = df[column].fillna(df[column].mode()[0])


df['car_age'] = 2023 - df['year']

df.drop(labels='year', axis=1, inplace=True)

df['fuel'] = df['fuel'].replace({
    'Gasoline':'gasoline',
    'Diesel': 'diesel',
    'Electric/Gasoline':'hybrid-petrol',
    'Electric/Diesel':'hybrid-diesel',
    'LPG': 'gas',
    'Electric':'electric'
    })

df['gear'] = df['gear'].replace({
    'Automatic':'automatic',
    'Manual':'mechanical',
    'Semi-automatic':'semi-automatic'
})

df = df.drop(labels=['make', 'model'], axis=1)

categorical_columns = ['gear', 'color', 'fuel', 'engine_has_gas', 'body_type', 'drivetrain']
encoded_df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df = encoded_df.astype(int)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#adding a new feature
df['mileage_per_year'] = df['mileage'] / df['car_age']

X = df.drop('price', axis=1)
y = df['price']

#division into training data (80%) and test data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#data scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#creating a model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

#model build
model.compile(loss='mean_squared_error', optimizer='adam')

#model training
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

#prediction on the test set
y_pred = model.predict(X_test_scaled)

#calculation of model evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R^2:", r2)

#converting predicted and actual values to DataFrame
results_df = pd.DataFrame({'Predicted': y_pred.flatten(), 'Actual': y_test.values})

#pint the table
print(results_df)

#select n first samples
n = 30
subset_df = results_df.iloc[:n]

subset_df.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Sample index')
plt.ylabel('Price')
plt.title('Comparison of predicted and actual prices for the first {} samples'.format(n))
plt.show()


