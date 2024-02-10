###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv("datasets/course_reviews.csv")
df.head()
df.shape

# Puanların dağılımına bakabiliriz
df["Rating"].value_counts()

# Soru sorulma dağılıma bakalım
df["Questions Asked"].value_counts()

# Sorulan soru kırılımında verilen puana bakalım
df.groupby("Questions Asked").agg({"Rating": ["count", "mean"]})



####################
# Average
####################

# Amacımız verilen puanlara göre bu ürünün "puanını" hesaplamak.
# Akla ilk gelen şey verilen puanların ortalamasını almaktır:
df["Rating"].mean()

# Fakat bu hesaplama ürünün satın alım süreciyle ilgili trendleri göz ardı etmektedir.
# Örneğin ürünün ilk çıktığı zamanlarda yarattığı memnuniyetle üzerinden bir yıl geçtikten sonra yarattığı memnuniyet
# değişmiş olabilir. Mesela zamanla daha iyi ürünler çıkmış olabilir ve bu ürüne olan beklentiler arttığı için zamanla
# alacağı puanlar düşmüş olabilir. Bu sebeple zaman göre ağırlıklı ortalama hesaplamak bu sorunu çözebilir



####################
# Time-Based Weighted Average
####################
# Puan Zamanlarına Göre Ağırlıklı Ortalama

df.head()
df.info()  # timestamp değişkeni object türünde. Bunu tarihe çevirmeliyiz

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Şimdi yorumların yapıldığı tarihlere göre kaç gün önce yapıldığı bilgisini hesaplamalıyız
# Bu analizi gerçekleştirdiğimiz tarihin şu olduğunu varsayalım
current_date = pd.to_datetime("2021-02-10")

df["days"]= (current_date - df["Timestamp"]).dt.days  # her bir yorumun kaç gün önce yapıldığını heaspladık
df["days"].describe().T

df[df["days"] <= 30]  # son 30 günde yapılan yorumlar

# Şimdi son 30 gün içinde verilen puanların ortalamasını hesaplayalım
df.loc[df["days"] <= 30, "Rating"].mean()

# 30-90 gün arasında verilan puanların ortalaması
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

# 90-180 arası
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

# 180'den büyük olanlar
df.loc[df["days"] > 180, "Rating"].mean()

# Şimdi bu ortalamalara belirli ağırlıklar verecek genel ortalamayı hesaplayalım
# Örmeğin periyotların ağırlıkları sırasıyla %28, %26, %24 ve % 22 olsun:

df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
df.loc[df["days"] > 180, "Rating"].mean() * 22/100

# Bu süreci fonksiyonlaştıralım
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["days"] > 180, "Rating"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)

# Puanın verildiği zaman periyoduna göre ağırlıklar belireyerek bu problemin üstesinden gelmiş olduk
# Şimdi şunu düşünmeliyiz: her kullanıcının verdiği yorumun etkisi aynı mı olmalıdır?



####################
# User-Based Weighted Average
####################

# Bu örnek özelinde "bu kursun hepsini takip eden kişi ile çok az bir miktarını
# takip eden kişinin puanı aynı ağırlıkta mı olmalıdır? Aynı olmadığını varsayalım.

df.groupby("Progress").agg({"Rating": ["count", "mean"]})  # kursta ilerleme kaydedenlerin verdiği puanlar daha yüksek

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22/100 + \
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24/100 + \
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26/100 + \
df.loc[df["Progress"] > 75, "Rating"].mean() * 28/100

# Bu süreci fonksyionlaştıralım
def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100

user_based_weighted_average(df)


####################
# Weighted Rating
####################


def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return (time_based_weighted_average(df) * time_w / 100 ) + (user_based_weighted_average(df) * user_w / 100)

course_weighted_rating(df)

course_weighted_rating(df,time_w=40, user_w=60)
