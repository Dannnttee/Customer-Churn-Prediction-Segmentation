import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv') #читаем файл

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') #тут превращаем ошибки в пустоту 
df = df.dropna() #Удаляем строки где есть пустота

df = df.drop('customerID', axis=1) #Удаляем кастомер айди, где аксис обозначает колонку, аксис 0 - строка

df_encoded = pd.get_dummies(df, drop_first=True) 

X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 # распределяем выборку между обучением и тестом

model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42) # параметры, 100 деревьев, глубина, сбалансированный вес ...?
model.fit(X_train, y_train) 


predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

X_clustering = df[['tenure', 'MonthlyCharges']]

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_clustering)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Cluster', palette='viridis')
plt.title('Сегментация клиентов: Стаж vs Платеж')
plt.show()