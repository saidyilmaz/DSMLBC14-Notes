##################################
# Customer Segmentation with RFM #
##################################

# RFM = Recency, Frequency, Monetary
# Bunlara RFM metrikleri denir: Recency = Yenilik, Frequency = Sıklık, Monetary: Parasallık
# Bu metrikleri kullanarak her bir müşteri için RFM skoru hesaplanır.
# Bu skorlar üzerinden segmentler oluşturacağız

# Bu süreci gerçeklertirmek için aşağıdaki adımları takip edeceğiz
# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 7. Tüm Sürecin Fonksiyonlaştırılması

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_ = pd.read_excel('Week 3 - CRM Analytics/datasets/online_retail_II.xlsx', sheet_name='Year 2009-2010')
df = df_.copy()
df.head()
df.shape
df.isnull().sum()

# essiz urun sayisi nedir?
df["Description"].nunique()

# her bir ürün kaç farklı faturada bulunmuş?
df["Description"].value_counts().head()

# her bir üründen toplamda kaç tane satılmış?
df.groupby("Description").agg({"Quantity": "sum"}).sort_values(by="Quantity", ascending=False).head()

# kaç tane fatura var?
df["Invoice"].nunique()

# fatura başına toplam kaç para kazanılmıştır?
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.groupby("Invoice").agg({"TotalPrice": "sum"})

###############################################################
# 3. Veri Hazırlama (Data Preparation)
###############################################################

df.shape
df.dropna(inplace=True)
df.describe().T

# Görüldüğü üzere Quantity ve TotalPrice değerlerinde negatif değerler var. Bunlar iadelere tekabül ediyor.
# Bunlardan kurtulmak için şunu yapıyoruz:
df = df[~df["Invoice"].str.contains("C", na=False)]

###############################################################
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

# Recency = analizin yapıldığı tarih - müşterinin son satın alım yaptığı tarih
# Frequency = satın alım sayısı
# Monetary = müşterinin bıraktığı toplam parasal değer

# Bu veri setinin içindeki verilerin son tarihinden iki gün sonra bu analizi yaptğımızı varsayalım
df["InvoiceDate"].max()
today_date = dt.datetime(2010, 12, 11)
type(today_date)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days,
                               "Invoice": lambda num: num.nunique(),
                               "TotalPrice": lambda totalprice: totalprice.sum()})
# "InvoiceDate": lambda date: (today_date - date.max()).days --> bu kısım
# her bir satır için bugünün tarihinden müşterinin alışveriş yaptığı son günün tarihini çıkarıp gün sayısını veriyor

# "Invoice": lambda num: num.nunique() --> bu kısımda kaç farklı fatura olduğunu gördük

# "TotalPrice": lambda totalprice: totalprice.sum() --> bu kısımda toplam ne kadar para harcadığını gördük

# columnlara Recency, Frequency, Monetary yazalım
rfm.head()
rfm.columns = ["Recency", "Frequency", "Monetary"]
rfm.head()

rfm.describe().T  # burada monetary değerinin min değerinin 0 olduğunu gözlemliyoruz. Bunları listeden çıkaralım

rfm = rfm[rfm["Monetary"] > 0]

rfm.shape  # böylece 4312 farklı müşteri için rfm metriklerini hesaplamış olduk

###############################################################
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
###############################################################

# receny skoru en düşük receny metriğine sahip olan müşteriye 5, en büyük olana 1 şeklinde 1-5 arası skorlar atar
# qcut listeyi küçükten büyüğe doğru sıralayıp bunu 5 eşit parçaya bölüp parçalara  listedeki labelları veriyor
rfm["recency_score"] = pd.qcut(x=rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])

# benzer durumu monetary değeri için yapalım
# en büyük monetary metriğine sahip olan müşteriye 5, en az olan müşteriye 1 değerini verelim
rfm["monetary_score"] = pd.qcut(x=rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])

# frequency için benzer şekilde ilerlersek bir hata ile karşılaşırız
### SORU: Bu hatanın sebebini de çözümünü de henüz anlayamadım...??? ###

rfm["frequency_score"] = pd.qcut(x=rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])

#artık frm skorlarını bir araya getirmeye hazırız

rfm["RFM_SCORE"] = (rfm["frequency_score"].astype(str) + rfm["recency_score"].astype(str))

# şampiyon müşteriler RFM skoru 55 olan müşterilerdir (frequency ve receny değerleri yüksek olanlar)
rfm[rfm["RFM_SCORE"] == "55"]  # "55" değerini string olarak yazdık çünkü tipi bu

###############################################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
###############################################################

# regex
# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',  # birinci ve ikinci elemanında 1 ya da 2 görürsen 'hibernating' diye isimlendir
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',  # birinci ve ikini elemanı 3 ise 'need_attention' diye isimlendir
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# yukarıdaki listeye göre müşterileri uygun segmentlere ayırmış olduk
rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)

# şimdi bu segmentleri analiz edelim
rfm[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])

# diyelim ki belli bir segmentteki müşterilerin bilgisini bir yere iletmek istiyoruz. Bu seçimi şu şekilde yapabiliriz:
rfm[rfm["segment"] == "new_customers"].index  # ilgili müşterilerin ID listesi

# elde ettiğimiz bu listeyi new_df isimli yeni bir dataframe içine aktardık
new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == 'new_customers'].index
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

# bu dataframe'i bir csv dosyası olarak kaydedelim
new_df.to_csv("new_customers.csv")

# aynı işlemi rfm'e de uygulayalım
rfm.to_csv("rfm.csv")


###############################################################
# 7. Tüm Sürecin Fonksiyonlaştırılması
###############################################################


def create_rfm(dataframe, csv=False):

    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))


    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

df = df_.copy()

rfm_new = create_rfm(df, csv=True)