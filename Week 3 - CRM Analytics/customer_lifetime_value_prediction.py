##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması


##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################
# burada kullanacağımız kütüphaneler için konsola "pip install lifetimes" komutunu girip ilgili kütüphaneleri indirmemiz gerekebilir

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: '%.4f' % x)

# veri setindeki aykırı değerleri baskılamak amacıyla aykırı değerler için gerekli
# sınırları aşağıdaki fonksiyon aracılığıyla hesaplayacağız
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# sınırlar belirlendikten sonra aykırı değerlerin yorumlarımızı etkilememesi için bunları baskılayacak
# aşağıdaki fonksiyonu kullanacağız
# bu fonksiyon belirlediğimiz sınırların altında/üstünde kalan değerleri sınır için belirlediğimiz değerlerle değiştirir
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
# not: yukarıda low_limit ile ilgili kısmı comment out etmemizin sebebi zaten veri setindeki negatif sayılardan
# kurtulmak için ayrı bir işlem uygulayacak olmamız. Uygulamayacak olsaydık bu basamağa ihtiyaç duyabilirdik.


#########################
# Verinin Okunması
#########################

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

#########################
# Veri Ön İşleme
#########################
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")  # "Quantity" ve "Price" değişkenlerinin aykırı değerlerini baskıladık
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]  # her bir ürünün toplam maliyetini hesapladık.

today_date = dt.datetime(2011, 12, 11)  # analiz yapılma gününü bu şekilde belirledik.

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# recency: Son satın alma üzerinden geçen zaman değil. İlk ve son satın alma arası... Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç  ### SORU: Neden toplam değil de ortalama yazılmış ???

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                                                         lambda x: (today_date - x.min()).days],
                                         "Invoice": lambda x: x.nunique(),
                                         "TotalPrice": lambda x: x.sum()})

# bu çıktının okunabilirliği az o yüzden bazı değişiklikler yapalım
cltv_df.columns = cltv_df.columns.droplevel(0)  # en üstteki levelı sildik
cltv_df.columns = ["recency", "T", "frequency", "monetary"]  # sutünlara ilgili isimlendiremeleri yaptık
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]  # her bir müşterinin fatura başına ortalama harcaması

cltv_df = cltv_df[cltv_df["frequency"] > 1]  # frequency değeri 1'den büyük olanları aldık

cltv_df["recency"] = cltv_df["recency"] / 7  # günlük değerleri haftalık değerlerle değiştirdik
cltv_df["T"] = cltv_df["T"] / 7


##############################################################
# 2. BG-NBD Modelinin Kurulması
##############################################################

# BG-NBD modelini kurmak için veri setimizin gamma ve beta dağılımında hangi katsayılara sahip olduğunu bulmamız gerek.
# önce modelimize bgf ismiyle modelimizi kuruyoruz ve gerekli parametreleri bu modele girerek uygun
# katsayıları elde ediyoruz.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])


################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1, cltv_df["frequency"], cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)

# ya da aşağıdaki fonksiyonu kullanabilirsiniz
bgf.predict(1, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]).sort_values(ascending=False).head(10)

# bunu veri setimize ekleyelim
cltv_df["expected_purc_1_week"] = bgf.predict(1, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])


################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.predict(4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]).sort_values(ascending=False).head(10)
cltv_df["expected_purc_1_month"] = bgf.predict(4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

# bir ayda şirkette beklenen satış sayısı:
bgf.predict(4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]).sum()

################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

cltv_df["expected_purc_3_month"] = bgf.predict(12, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])
bgf.predict(12, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]).sum()

################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

# GG modelini kurmak için veri setimizin gamma ve beta dağılımında hangi katsayılara sahip olduğunu bulmamız gerek.
# önce modelimize bgf ismiyle modelimizi kuruyoruz ve gerekli parametreleri bu modele girerek uygun
# katsayıları elde ediyoruz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

# bunu veri setimize ekleyelim
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3,  # 3 aylık
                                   freq='W',  # T'nin frekans bilgisi yani haftalık
                                   discount_rate=0.01)

cltv = cltv.reset_index()

# yeni değerleri ilgili metriklerin olduğu tabloyla birleştirereik her şeyi bir arada görüyoruz
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

# bu tabloyu clv değerleri azalacak şekilde sıralarsak şirket için en büyük potansiyele sahip müşterileri bulmuş oluruz
cltv_final.sort_values(by="clv", ascending=False)


##############################################################
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
##############################################################

cltv_final["segment"] = pd.qcut(cltv["clv"], q=4, labels=["D", "C", "B", "A"])

cltv_final.groupby("segment").agg({"count", "mean", "sum"})


##############################################################
# 6. Çalışmanın Fonksiyonlaştırılması
##############################################################

def create_cltv_p(dataframe, month=3):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final


df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")
