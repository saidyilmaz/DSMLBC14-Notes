#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi ve averagebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

### Gerekli kütüphanelerin yüklenmesi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz.
# Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
df_control = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
df_test = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")



# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

# Veri setlerindeki sayılar float formatında. Bunları integera çevirmek gerekiyor
df_control = df_control.astype(int)
df_test = df_test.astype(int)

df_control.describe().T
df_test.describe().T


# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
df_test.columns = [col + "_test" for col in list(df_test.columns)]
df_control.columns = [col + "_control" for col in list(df_control.columns)]

df_all = pd.concat([df_test, df_control], axis=1)



#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

# HO: M1 = M2
# Test ve Control gruplarının "Purchase" ortalamaları arasında istatistiksel anlamda bir fark yoktur

# H1: M1 != M2


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

df_all["Purchase_control"].mean()
df_all["Purchase_test"].mean()

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

# NORMALLİK VARSAYIMI:
# H0: normal dağılım varsayımı sağlanmaktadır.
# H1: normal dağılım varsayımı sağlanmamaktadır.

# kontrol gurubu için:
test_stat, pvalue = shapiro(df_all["Purchase_control"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # p-value 0.05'ten büyük olduğu için H0 reddedilemez.


# test grubu için:
test_stat, pvalue = shapiro(df_all["Purchase_test"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # p-value 0.05'ten büyük olduğu için H0 reddedilemez.

# Sonuç: Kontrol ve test gruplarının normal dağılıma sahip olduğunu varsayabiliriz.

# VARYASN HOMOJENLİĞİ VARSAYIMI
# HO: Varyans homojenliği varsayımı sağlanmaktadır.
# H1: Varyans homojenliği varsayımı sağlanmamaktadır.

test_stat, pvalue = levene(df_all["Purchase_control"], df_all["Purchase_test"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # p-value 0.05'ten büyük olduğu için H0 reddedilemez.

# Sonuç: Varyans homojenliği varsayımı yapılabilir.

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

# Normallik ve varyans homojenliği varsayımlarını ikisi de geçerli olduğu için Bağımsız İki Örneklem T Testi uygulanmalı

test_stat, pvalue = ttest_ind(df_all["Purchase_control"], df_all["Purchase_test"], equal_var= True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# p-value 0.05'ten büyük olduğu için H0 reddedilemez.
# Yani kontrol ve test gruplarının satın alma ortalamaları arasında istatiksel açıdan anlamlı bir fark yoktur.



##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# Her iki örneklemin dağılımının da normal dağılımda olduğunu varsayabildiğimiz
# için Bağımsız İki Örneklem T Testi kullanmamız gerekti.
# Varyasnları homojenliğinin var olduğunu varsayabildiğimiz için teste parametre olarak
# equal_var = True argümanını girdik.



# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# Sonuç olarak gözlemlediğimiz ver setinde kontrol ve test grubunun ortalamalarının farklı çıkmalarının sebebi
# %5 hata payıyla tesadüfen gerçekleşmiştir. Aslında bu iki grubun ortalamaları arasında (%5 hata payıyla)
# istatistiksel bir farklılık yoktur.

# Burada "average bidding"in "maximum bidding"e kıyasla istatistiksel açıdan anlamlı bir fark yaratmadığı
# %95 anlamlılık düzeyinde söylenebilir.
