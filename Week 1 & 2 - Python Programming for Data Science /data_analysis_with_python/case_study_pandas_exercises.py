###############################################
#        CASE STUDY - PANDAS EXERCISES        #
###############################################

#Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

#Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.

df["sex"].value_counts()

#Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.

df.nunique()

#Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.

df["pclass"].nunique()

#Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

df[["pclass", "parch"]].nunique()

#Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz. Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.

df["embarked"].dtypes
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtypes

#Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.

df[df["embarked"] == "C"].head()

#Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.

df[df["embarked"] != "S"].head()

#Görev9: Yaşı 30dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df[(df["age"] < 30) & (df["sex"] == "female")].head()

#Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.

df[(df["age"] > 70) | (df["fare"] > 500)].head()

#Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()

#Görev 12: who değişkenini dataframe’den çıkarınız.

df = df[df.columns.drop("who")]
# ya da
df = df.drop("who", axis=1)

#Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.

m = df["deck"].mode()
for index in df["deck"][df["deck"].isnull()].index:  #değeri boş olan indexlerin listesi üzerinde dönecek bir döngü
    df["deck"][index] = m
# ya da
df["deck"].fillna(df["deck"].mode().iloc[0])  ###SORU: iloc[0] ne işe yaradı burada

#Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.

df["age"].fillna(df["age"].median())

#Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})

#Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)

def is_less_than_30(age):
    if age < 30:
        return 1
    else:
        return 0
df["age_flag"] = [is_less_than_30(value) for value in df["age"]]

# ya da

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

#Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

df = sns.load_dataset("tips")

#Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz. Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})

#Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df.groupby(["day","time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

#Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz. Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

new_df = df[(df["time"] == "Lunch") & (df["sex"] == "Female")]

new_df.groupby(["day"]).agg({"total_bill": ["sum", "min", "max", "mean"],
                             "tip": ["sum", "min", "max", "mean"]})

#Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

index_list = [i for i in df.index if df["size"][i] < 3 and df["total_bill"][i] >10]
df.loc[index_list, "total_bill"].mean()

#Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.

df["total_bill_tip_sum"] = [df["total_bill"][i] + df["tip"][i] for i in df.index]

# yada

df["total_bill_tip_sum"] = df["total_bill"] + df["sum"]

#Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.

df = df.sort_values(by = ["total_bill_tip_sum"], ascending = False)
new_df = df.reset_index(drop= True)
new_df = new_df[0:30]
