#######################################################
# KEŞİFÇİ VERİ ANALİZİ VE KURAL TABANLI SINIFLANDIRMA #
#######################################################

#############################################
# İş Problemi
#############################################
# Bir seyahat acentası İstanbul'un çeşitli semtlerinde bulunun ve çeşitli tipteki ev/odaları uygun
# profillere ayırmak istemektedir. Böylece bu profillere göre yeni bir evin/odanın fiyatının ne olması gerektiğine dair
# fikir edinebileceklerdir.

# Örneğin: Beyoğlu'nda bulunan bir otel odasının fiyatlandırmasının ortalama ne olması gerektiği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# listings_airbnb.csv dosyasında Airbnb'den 29 Aralık 2023'te alınan geçmiş yıllara yönelik İstanbul'daki
# ilanların bilgileri bulunmaktadır.

# id: ilanın kimlik numarası
# name: ilanın ismi
# host_id: ilanı veren ev sahibinin kimlik numarası
# host_name: ilanı veren ev sahibinin ismi
# neighbourhood_group: evin bulunduğu semtin dahil olduğu grup
# neighbourhood: evin bulunduğu semt
# latitude: evin bulunduğu enlem
# longitude: evin bulunduğu boylam
# room_type: kiralanan yerin türü
# price: ilanın fiyatı
# minimum_nights: evin tutulduğu minimum gece sayısı
# number_of_review: ilana yapılan yorum sayısı
# reviews_per_month: ilana yapılan aylık yorum sayısı
# availability_365: ilandaki evin yılın kaç günü kiralamaya uygun olduğu
# license: ilanın lisans numarası


#####################################
# GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ #
#####################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df = pd.read_csv("datasets/listings_airbnb.csv")



##########################################
## VERİYE İLK BAKIŞ VE VERİ ÖNHAZIRLIĞI ##
##########################################
# veri setinin ilk 5 satırı
df.head()

# veri setindeki değişken ve satır sayısı
df.shape

# veri setindeki dğeişkenlerin isimleri
df.columns

#veri setindeki değişkenlerde kaç tane boş değer var?
df.isnull().sum()

# "neighbourhood_group" ve "licence" değişkenleri tamamen boş olduğu için onları veri setinden silebiliriz
df.drop(["neighbourhood_group", "license"], axis=1, inplace=True)

# "neighbourhood" değişkeninde hangi eşsiz değerler olduğunu şu şekilde görebiliriz
df["neighbourhood"].unique()

# "neighbourhood" değşkenindeki eşsiz değerlerin sayısı:
df["neighbourhood"].nunique()

# her bir eşsiz değerden kaç tane olduğu:
df["neighbourhood"].value_counts()

# "neighbourhood" değişkenindeki değerlerin sayısını bir grafik üzerinde görmek istersek şunu yapabiliriz
df["neighbourhood"].value_counts().plot(kind='bar')
plt.show()

# benzer gözlemleri "room_type değişkenine de yapabiliriz
df["room_type"].unique()
df["room_type"].value_counts()

# bu değişkenin dağılımını da bir grafik ile görselleştirebiliriz
sns.countplot(y=df["room_type"], data=df)
plt.show()

# "price değişkeninin aldığı en büyük değeri, aldığı değerlerin ortalamasını görmek için
df["price"].mean()
df["price"].max()

# yukarıdaki gözlemleri ve daha fazlasını görebilmek için "describe()" metodunu kullanabiliriz
df["price"].describe()

# bunu veri setindeki bütün sayısal değişkenlere yapmak mümkün
df.describe()

# veri setinde boş değerler var bunu "df.isnull().sum()" satırında gözlemlemiştik.
# boşluk içeren satırları silerek daha detaylı analizler yapmaya başlayabiliriz
df.dropna(inplace=True)

# veri setindeki fiyatların oda türüne göre ortalamalarını, en büyük ve en küçük değelerini görebilmek için
df.groupby(["room_type"]).agg({"price": ["mean", "max", "min"]})

# aynı gözlemi "neighbourhood" özelinde yapalım
df.groupby(["neighbourhood"]).agg({"price": ["mean", "max", "min"]})

# Her bir bölgedeki bütün oda türlerinin ortalama fiyatlarını öğrenmek için şunu yapabiliriz:
df.groupby(["room_type", "neighbourhood"]).agg({"price": "mean"})

# Eğer "Beyoglu"ndaki odaların fiyatlarının oda türlerine göre analizini görmek istiyorsak:
df[df["neighbourhood"] == "Beyoglu"].groupby(["room_type"]).agg({"price": ["mean", "max", "min"]})

# Aynı analizi "Sisli" için de yapabiliriz
df[df["neighbourhood"] == "Sisli"].groupby(["room_type"]).agg({"price": ["mean", "max", "min"]})

# Veri setimizi fiyatlar azalacak şekilde sıralayabiliriz
df = df.sort_values(by="price",ascending=False)

# DİKKAT: Artık veri setindeki satırların yeri değiştiği için her bir satırların indexlerinin yeri de değişti

# bu indexleri düzeltmek için:
df = df.reset_index(drop= True)



############################
## SEGMENTASYON İŞLEMLERİ ##
############################

# Şimdi her bir kaydı bulunduğu semte ve oda türüne göre kategorize etmeye çalışalım
# öncelikle verilen oda türüne karşılık gelecek bir etiket yaratacak fonksiyonu yazalım:
df["room_type"].unique()  # odalar 4 farklı şekilde isimlendirilmiş
def create_room_tag(type_of_room):
    if type_of_room == "Entire home/apt":
        return "entire_home"
    elif type_of_room == "Private room":
        return "private_room"
    elif type_of_room == "Hotel room":
        return "hotel_room"
    elif type_of_room == "Shared room":
        return "shared_room"
    else:
        return "undefined_room_type"

# şimdi bu fonksiyonu bütün satırlara uygulayarak veri setimize "room_tag" isimli yeni bir sütun ekleyelim
df["room_tag"] = [create_room_tag(row) for row in df["room_type"]]
df["room_tag"].unique()  # artık her bir kayıtta oda türüne ait room_tag isimli etiketler mevcut

# her kayda "neighbourhood_room_type" formatında bir etiket verelim
df["neighbourhood_room_type_tag"] = df["neighbourhood"] + "_" + df["room_tag"]

# Artık elimizdeki listeyi kullanarak her bir semt veoda türüne göre kaç tane kayıt var, fiyatların max ve min
# değerleri neler, fiyatların ortalamaları kaç gibi sorularu cevaplayabiliriz
# Bu bilgileri ayrı bir veri setine kaydedelim
new_df = df.groupby("neighbourhood_room_type_tag").agg({"price": ["count", "max", "mean", "min"]})

# Eğer Bayoglu'da kaç tane otel odası var, bunların fiyatlarının maksimum-minimum değerleri ve ortalamaları
# nelerdir sorularına cevap arıyorsak şunu yapabiliriz:
new_df.loc["Beyoglu_hotel_room"]

# Örneğin Uskudar'daki özel odaların bilgileri şu şekildedir:
new_df.loc["Uskudar_private_room"]



################################
## VERİLERİN DIŞA AKTARILMASI ##
################################

# Gözlemleneceği üzere bu veri setinde kayıtların etiketleri index olarak bulunuyor.
# Bu etiketleri de bir değişken olarak (yani veri setine bir sütun olarak) ekleyebiliriz
new_df.reset_index(inplace=True)

# Fakat görüleceği üzere bu veri setinin okunabilirliği düşük. Sütunları yeniden adlandırarak
# veri setini daha okunabilir hale getirebiliriz.
new_df.columns = ["neighbourhood_room_type_tag", "number_of_listings", "max_price", "average_price", "min_price"]

# Artık bu veri setini dışa .csv dosyası olarak aktarabiliriz:
new_df.to_csv("segmentation_based_data_analysis.csv")


###################################
## SÜRECİN FONKSİYONLAŞTIRILMASI ##
###################################
def create_room_tag(type_of_room):
    if type_of_room == "Entire home/apt":
        return "entire_home"
    elif type_of_room == "Private room":
        return "private_room"
    elif type_of_room == "Hotel room":
        return "hotel_room"
    elif type_of_room == "Shared room":
        return "shared_room"
    else:
        return "undefined_room_type"

def create_segments(dataframe):

    # Veri önhazırlığı
    if "neighbourhood_group" in dataframe.columns:
        dataframe.drop(["neighbourhood_group"], axis=1, inplace=True)

    if "license" in dataframe.columns:
        dataframe.drop(["license"], axis=1, inplace=True)

    dataframe.dropna(inplace=True)
    dataframe = dataframe.sort_values(by="price", ascending=False)
    dataframe = dataframe.reset_index(drop=True)

    # Segmentasyon
    dataframe["room_tag"] = [create_room_tag(row) for row in dataframe["room_type"]]
    dataframe["neighbourhood_room_type_tag"] = dataframe["neighbourhood"] + "_" + dataframe["room_tag"]
    new_df = dataframe.groupby("neighbourhood_room_type_tag").agg({"price": ["count", "max", "mean", "min"]})

    # Dışa aktarma
    new_df.reset_index(inplace=True)
    new_df.columns = ["neighbourhood_room_type_tag", "number_of_listings", "max_price", "average_price", "min_price"]
    new_df.to_csv("segmentation_based_data_analysis.csv")

    return new_df


### What else could have been done with this data set?
# The minimum stay, price and number of reviews have been used to estimate the the number of nights booked and
# the income for each listing, for the last 12 months.
#
# Is the home, apartment or room rented frequently and displacing units of housing and residents?
#
# Does the income from Airbnb incentivise short-term rentals vs long-term housing?

# The housing policies of cities and towns can be restrictive of short-term rentals, to protect housing for residents.
#
# By looking at the "minimum nights" setting for listings, we can see if the market has shifted to longer-term stays.
# Was it to avoid regulations, or in response to changes in travel demands?
#
# In some cases, Airbnb has moved large numbers of their listings to longer-stays to avoid short-term rental
# regulations and accountability.

# Some Airbnb hosts have multiple listings.
#
# A host may list separate rooms in the same apartment, or multiple apartments or homes available in their entirity.
#
# Hosts with multiple listings are more likely to be running a business, are unlikely to be living in the property,
# and in violation of most short term rental laws designed to protect residential housing.
