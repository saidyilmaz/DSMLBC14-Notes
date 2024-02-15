############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################
import numpy as np
# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# 1. Veri Ön İşleme
############################################

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
# pip install openpyxl
# df_ = pd.read_excel("datasets/online_retail_II.xlsx",
#                     sheet_name="Year 2010-2011", engine="openpyxl")


df.describe().T
df.isnull().sum()
df.shape


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T


############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

df.head()

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1


# Gözlemlerimizi tek bir ülkeye sınırlayalım
df_fr = df[df['Country'] == "France"]

# Invoice ve Description üzerinden groupby yapıp quantitylerin toplamına bakalım
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"})

# Her bir invioce için hangi üründen kaç tane olduğunu listeledik fakar bu ürünlerin
# sütunlarda değişken olarak var olmasını sağlayalım
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack()

# Her bir boşluğa 0 değerini koyalım
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0)

# Sıfırdan farklı olan her bir değeri 1 ile değiştirelim
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}). \
    unstack().fillna(0).applymap(lambda x: 1 if x >0 else 0)

# Artık bu süreci fonksiyonlaştırmaya hazırız:
# Bu işlemleri Description üzerinden de StockCode üzerinden yapabiliriz, bunu fonksiyona bir özellik olarak tanımlayalım
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)  # Decription kullanmak için

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)  # StockCode kullanmak için

fr_inv_pro_df = fr_inv_pro_df.astype('bool')  # burada hesaplamaları daha hızlı yapmak için veri yapısını
                                                # bool'a çevirerek boyutunu küçültebiliriz, opsiyonel

# İleride hangi StockCode'un hangi ürüne denk geldiğini öğrenmek isteyebiliriz. Bunu cevaplayacak fonksiyon:
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

# Örneğin 10120 kodlu ürünün ismini öğrenmek istiyorsak:
check_id(df_fr, 10120)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################

# apriori fonksiyonunu kullanarak her bir ürünün ve ürün kümesinin 0.02 eşik değeriyle supportunu hesaplıyoruz
frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.02,
                            use_colnames=True,
                            low_memory=True)

frequent_itemsets.sort_values("support", ascending=False)

# Hesapladığımız supportları kullanarak ihtiyacımız olan birliktelik kurallarını hesaplıyoruz
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.02)

# Örneğin supportu 0.05'ten büyük olan, confidenceı 0.1'den büyük olan lifti 5'ten olan ürünlerin listesine bakalım:
rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)]

check_id(df_fr, 21086)

# yukarıda baktığımız ürünleri confidence'ı artacak şekilde sıralayarak gözlem yapalım
rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)]. \
sort_values("confidence", ascending=False)

############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################

# Burada şu ana kadar yapmış olduğumuz bütün işlemleri fonksiyonlaştırıyoruz

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.02)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek:
# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id)

# Örneğin lift değerleri artacak şekilde verileri sıralayalım
sorted_rules = rules.sort_values("lift", ascending=False)

# tavsiye edilecek ürünleri içine koyacağımız boş bir liste tanımlayalım
recommendation_list = []

# lift değerine göre en yüksek değere sahip olan consequent ürünü bizim için en tavsiye edilesi ürün
# olacağından bunu listeye ekleyelim (bu seçimi lift üzerinden yapmak zorunda değildik,
# ihtiyaca göre confidence da kullanılabilirdi)
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

# yukarıdaki döngü sayesinde tavsiye edilecek ürünleri sırayla bu listeye eklemiş olduk
len(recommendation_list)
recommendation_list[0:3]

# ürnün ne olduğunu merak ediyorsak bakabiliriz:
check_id(df, 22561)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)
