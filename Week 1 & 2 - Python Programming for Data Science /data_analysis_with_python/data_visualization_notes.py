#####################################################
#     VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN     #
#####################################################

##################
### MATPLOTLIB ###
##################

# matplotlib veri görselleştirmesinin atası sayılabilecek son derece ilkel (low level) bir görselleştirme kütühanesidir.

# Kategorik değişkenler sütun grafiği ile görselleştirilir. countplat ya da bar kullanabiliriz.
# Sayısal değişkenler histogram ya da kutu grafiği kullanabiliriz. boxplot kullanabiliriz.

#######################################
## KATEGORİK DEĞiŞKEN GÖRSELlEŞTİRME ##
#######################################

import pandas as pd # ihtiyacımız olan kütüphaneleri import ediyoruz
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None) #bütün sütunların görünmesini sağlıyoruz
pd.set_option('display.width', 500) #bütün sütunların aynı satırda görünmesini sağlıyoruz
df = sns.load_dataset("titanic") # "titanic" isimli veri setini "df" isimli bir dataframe içerisine alıyoruz
df.head() # veri setinin ilk 5 satırını görüntülüyoruz

df["sex"].value_counts() #sex değişkeninine göre veri setimizdeki elemanların sayısını görüntüledik
df["sex"].value_counts().plot(kind='bar') #yukarıdaki verileri 'bar' tipi bir grafiğe dönüştürdük
plt.show() #oluşturduğumuz grafiği görüntüledik

### SORU: Sütunların yerini nasıl değiştirebiliriz??? ###

## bu aşamada grafiği görüntüleyemiyorsak matplotlib kütüphanesi güncel olmayabailir
## güncellemek için "pip install matplotlib" ya da "pip install --upgrade matplotlib" komutunu kullanabiliriz

#######################################
##  SAYISAL DEĞiŞKEN GÖRSELlEŞTİRME  ##
#######################################

plt.hist(df["age"]) #"age" değişkenininin aldığı değerleri histogram grafiğine döküyoruz
plt.show() #grafiği görüntülüyoruz

### SORU: Yaş aralıklarını nasıl değiştirebiliriz???
### SORU: Sütunların arasına nasıl boşluk koyabiliriz???

plt.boxplot(df["fare"]) #"fare" değişkeninin aldığı değerleri bir kutu grafiğine döktük
plt.show() #grafiği görüntüledik

##############################
##  MATPLOTLIB ÖZELLİKLERİ  ##
##############################

##  plot özelliği

x = np.array([1, 3, 8])
y = np.array([0, 7, 150])

plt.plot(x, y) #apsisleri x değerleri ordinatları y değerleri olan noktaları bir çizgi ile birleştiren bir grafik oluşturur
plt.show() # grafiği görüntüler

plt.plot(x, y, 'o') #noktaların yerini çizgi kullanmadan büyük noktalarla belirtmesini sağlamak için 'o' argümanını ekledik
plt.show()

##  marker özelliği

z = np.array([13, 28, 11, 100])

plt.plot(z, marker='s') #çizgi grafiği üzerinde ilgili noktaları 'o' sembolü ile işaretler. '*' gibi farklı işaretler kullabılabilir
plt.show()

### SORU: 'o' ve '*' dışında başka hangi markerlar vardır??? ###
### CEVAP: markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's", 'D', "d", 'p', 'H', 'h'] ###

##  line özelliği

z = np.array([13, 28, 11, 100])

plt.plot(z, linestyle="dashed") #line grafiği oluşturur ama çizginin stilini dashed olarak ayarlar. Diğer stiller "dotted" ve "dashdot"
plt.show()

plt.plot(z, linestyle="dashdot", color="r") #çizginin rengini değiştirmek için "color" argümanı eklenir
plt.show()

##  multiple line özelliği

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])

plt.plot(x) #bu şekilde iki çizgi de aynı koordinat sistmein de görüntülenir
plt.plot(y)
plt.show()

##  labels özelliği

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x,y)

plt.title("Bu ana başlık") #grafiğie başlık yazar
plt.xlabel("Bu x ekseni") #x eksenine isim verir
plt.ylabel("Bu y ekseni") #y eksenine isim verir
plt.grid() #grafiğe grid ekleyerek okunaklılığını arttırır
plt.show()

##  subplots özelliği

# 3 grafiği bir satırda ve üç sütunda olacak şekilde görüntüleyeceğiz
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1) # açılacak pencerede 1 satır ve 3 sütundan oluşacak şekilde 3 tane grafik görüneceğini ve
                     # ve bu grafiklerden birincisini oluşturacağımızı belirtiyoruz.
plt.title("1")
plt.plot(x,y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 3, 2) # ikinci grafiği oluşturuyoruz
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3) # üçüncü grafiği oluşturuyoruz.
plt.title("3")
plt.plot(x, y)

###################
###   SEABORN   ###
###################

# seaborn veri görselleştirme işlemleri için kullanılabilecek high level kütühanedir

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None) #bütün sütunların görünmesini sağladık
pd.set_option('display.width', 500) #bütün sütunların aynı satırda görünmesini sağladık
df = sns.load_dataset("tips") # "tips" isimli veri setini aldık
df.head()

## Kategorik Değişken Görselleştirme ##

df["sex"].value_counts() # "sex" değişkeninin veri dağılımını gösterir
sns.countplot(x=df["sex"], data=df) # "df" içindeki datadan "sex" değişkenini x ekseninde göstermesi için countplot komutunu kullandık
plt.show()

## Sayısal Değişken Görselleştirme ##

sns.boxplot(x=df["total_bill"]) # boxplot komutu ile "total_bill" değişkenini x ekseninde kutu grafiği ile görüntüledik
plt.show()

df["total_bill"].hist() # "total_bill" değişkenini histogram grafiğin kullanarak görselleştirdik
plt.show()

### SORU: Neden bu iki grafik şeklinin syntaxları farklı??? ###
