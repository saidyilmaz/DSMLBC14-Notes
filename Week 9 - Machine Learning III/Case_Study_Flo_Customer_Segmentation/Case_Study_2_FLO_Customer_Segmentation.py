
import numpy as np
import pandas as pd
import random
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

### GÖREV 1: VERİYİ HAZIRLAMA ###

### Adım 1: Veriyi okutunuz.

df = pd.read_csv("Week 9 - Machine Learning III/Case_Study_Flo_Customer_Segmentation/flo_data_20k.csv")

### Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz. Yeni değişkenler türetiniz

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

df["Recency"] = [(today_date - date).days for date in df["last_order_date"]]
df["Tenure"] = df["first_order_date"].apply(lambda x: (today_date - x).days)
df["Average_Monetary"] = df["customer_total_value"] / df["order_num_total"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """

    Veri setindeki kategorik değşkenler için one hot encoding işlemini yapar

    Parameters
    ----------
    dataframe : Veri setini ifade eder
    categorical_cols : Kategorik değişkenleri ifade eder
    drop_first : Dummy değişken tuzağına düşmemek için ilk değişkeni siler

    Returns
    -------
    One-hot encoding işlemi yapılmış bir şekilde "dataframe"i return eder

    Notes
    -------
    Fonksiyonun "pandas" kütüphanesine bağımlılığı bulunmaktadır.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# df = one_hot_encoder(df, ["order_channel"])

df.columns

var_list = ['order_num_total_ever_online', 'order_num_total_ever_offline', 'customer_value_total_ever_offline',
            'customer_value_total_ever_online', 'order_num_total',
            'customer_total_value', 'Recency', 'Tenure', 'Average_Monetary']

'''
var_list = ['order_num_total_ever_online', 'order_num_total_ever_offline', 'customer_value_total_ever_offline',
            'customer_value_total_ever_online', 'order_num_total',
            'customer_total_value', 'Recency', 'Tenure', 'Average_Monetary', 'order_channel_Desktop',
            'order_channel_Ios App', 'order_channel_Mobile']
'''

### GÖREV 2: K-MEANS İLE MÜŞTERİ SEGMENTASYONU ###

### Adım 1: Değişkenleri standartlaştırınız.

df_copy = df

df = df[var_list]

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

### Adım 2: Optimum küme sayısını belirleyiniz.

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_  # optimum küme sayısı 6.

### Adım 3: Model ve segmentleri oluşturma

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters_kmeans = kmeans.labels_

df = df_copy
df["cluster"] = clusters_kmeans
df["cluster"] = df["cluster"] + 1

var_list.append("cluster")

df[var_list].groupby("cluster").agg(["count","min", "mean", "max"])


### GÖREV 3: HIERARCHICAL CLUSTERING İLE MÜŞTERİ SEGMENTASYONU ###

### Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

df = df_copy

df = df[var_list]

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, "complete")


plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dend = dendrogram(hc_average, truncate_mode="lastp", p=10, show_contracted=True, leaf_font_size=10)
plt.show()

### Adım 2: Modelinizi oluşturup müşterileri segmentleyiniz.

cluster = AgglomerativeClustering(n_clusters=4, linkage="average")

clusters = cluster.fit_predict(df)

df = df_copy

df["hi_cluster"] = clusters

df["hi_cluster"] = df["hi_cluster"] + 1


### Adım 3: Her bir segmenti istatistiksel olarak inceleyiniz.


var_list.append("hi_cluster")

df[var_list].groupby("hi_cluster").agg(["count", "min", "mean", "max"])

