import numpy as np 
import pandas as pd 
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize']=12,8
import matplotlib.pyplot as pl 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('avocado.csv')
print(data.head())

data['Date'] = pd.to_datetime(data['Date'])

pl.figure(figsize=(12,8))
pl.title("Price Distribution")
sns.distplot(data['AveragePrice'], color='r')
pl.show()

sns.boxplot('AveragePrice', 'type', data=data, palette='pink')
pl.show()

mask = data['type'] == 'conventional'
order = (
        data[mask & (data['year']==2018)]
        .groupby('region')['AveragePrice']
        .mean()
        .sort_values()
        .index
)

sns.factorplot('AveragePrice', 'region', data=data[mask], size=13, aspect=0.8, hue='year', palette='magma', join=False, order=order)
pl.show()


mask2 = data['type'] == 'organic'
order = (
        data[mask2 & (data['year']==2018)]
        .groupby('region')['AveragePrice']
        .mean()
        .sort_values()
        .index
)

sns.factorplot('AveragePrice', 'region', data=data[mask2], size=13, aspect=0.8, hue='year', palette='magma', join=False, order=order)
pl.show()

data['Month'] = data['Date'].dt.month

regions = ['PhoenixTucson', 'Chicago', 'NewYork']

mask3 = (
        data['region'].isin(regions)
        & (data['type']=='conventional')  
)

sns.factorplot('Month', 'AveragePrice', data=data[mask3], hue='year', row='region', aspect=2, palette='Blues')
pl.show()

label = LabelEncoder()
label.fit(data.type.drop_duplicates())
data.type= label.transform(data.type)

cols = ['AveragePrice', 'type', 'year', 'Total Volume', 'Total Bags']
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.7)
hm = sns.heatmap(cm, cbar=True,annot=True,square=True, fmt='.2f',annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
pl.show()