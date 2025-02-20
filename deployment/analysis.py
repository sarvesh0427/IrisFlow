from sklearn.datasets import load_iris
import pandas as pd


data = load_iris()
x= data.data
y= data.feature_names
df = pd.DataFrame(x,columns=y)
#
# df['target'] = data.target
# print(df.head())
# print(df['petal length (cm)'].max())
# print(df['petal length (cm)'].min())
# print(df['petal length (cm)'].mean())
# print(data.target)
# print(data.target_names)
# print(df.head())

print(df)