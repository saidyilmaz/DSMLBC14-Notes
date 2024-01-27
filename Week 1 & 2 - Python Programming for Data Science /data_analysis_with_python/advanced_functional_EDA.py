#######################################################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ - ADVANCED FUNCTIONAL EDA #
#######################################################################

# AMAÇ: Hızlı bir şekilde genel fonksiyonlarla elimizdeki veriyi analiz etmek

##################
# 1. Genel Resim #
##################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df.head()  # veri setinin ilk 5 satırını gösterir
df.tail()  # veri setinin son iki satırını gösterir
df.shape  # veri setinin satır ve sütun sayısını verir
df.info()  # veri setindeki değişkenlerin aldığı non-null değerlerin sayısını ve tiplerini verir
df.columns  # sütunlarda bulunan değişkenlerin isimlerini bir liste içerisinde verir
df.index  # veri setinin insexleri hakkında bilgi verir
df.describe().T  # sayısal değişkenlerin ortalama, standart sapma gibi betimsel değerlerini verir
# rahat okunması için .T komutuyla veri setinin transpozunu aldık
df.isnull().values.any()  # veri setinde eksiklik olup olmadığını görmek için kullanıyoruz
# df.isnull() bize dolu her değer için false boş her değer için true
# değerine sahip yeni bir dataframe verir
# .values bu dataframedeki her bir satırı eleman olarak barındıran bir list oluşturur
# .any() bu listede true değeri (yani boş veri) olup olmadığına bakar
# Böylece veri setinde eksiklik olup olmadığını öğrenmiş oluruz
df.isnull().sum()  # df.isnull() veri setindeki true/false (yani 1/0) değerlerini toplayarak


# veri setinde hangi değişkende kaç tane boşluk olduğunu söyler

## Şimdi veri setini girdiğimizde bize veri setiyle ilgili genel bilgileri verecek bir fonksiyon tanımlayacağız

def check_df(dataframe, head=5):  # parametre olarak dataframe ve 5 değerine sabitlenmiş bir head değişkeni tanımladık
    print("##################### Shape #####################")
    print(dataframe.shape)  # veri setinin shape bilgisini ekrana yazar
    print("##################### Types #####################")
    print(dataframe.dtypes)  # veri setindeki her bir değişkenin değişken türlerini ekrana yazar
    print("##################### Head #####################")
    print(dataframe.head(head))  # veri setinin ilk head (bu örnekte 5) satırını ekrana yazar
    print("##################### Tail #####################")
    print(dataframe.tail(head))  # veri setinin son head (bu örnekte 5) satırını ekrana yazar
    print("##################### NA #####################")
    print(dataframe.isnull().sum())  # veri setindeki değişkenlerde bulunan eksik verilerin sayısını verir
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)  # sayısal değişkenlerin dağılım bilgisini ekrana yazar


check_df(df)  # yazdığımız fonksiyona "df" veri setini koyarak çıktıyı inceliyoruz

df2 = sns.load_dataset("tips")  # başka bir veri seti kullanarak fonksiyonun çıktısını gözlemliyoruz
check_df(df2)

### SORU: describe içine koyduğumuz liste ne işe yarıyor??? ###


####################################################################
# 2 Kategorik Değişken Analizi - Analysis of Categorical Variables #
####################################################################


df["embarked"].value_counts()  # embarked değişkenindeki verilerin hangisinden kaç tane olduğunu gösterir
df["sex"].unique()  # sex değişkeninde birbirinden farklı kaç çeşit değer varsa onların listesini verir
df["sex"].nunique()  # sex değişkeninde kaç tane birbirinden farklı değer olduğunu gösterir

## Şimdi veri setimizdeki değişkenlerden kategorik olanları bir liste içerisinde
## toplamak için list comprehension yapısı kullanacağız


cat_cols = [col for col in df.columns if str(df[col].dtype) in ['object', 'bool', 'category']]
# df[col].dtype bize col değişkeninin türünü verir, str fonksiyonu ile bu bilgiyi stringe çeviririz.
# eğer col değişkeninin türü yukarıda verilen listede var ise bunu cat_cols listesine eklemesini sağladık
# böylece tipi object, bool ya da category olan değişkenleri bir listeye eklemiş olduk.


## Şimdi numerik değerlere sahip olmasına rağmen aslında kategorik olan değişkenleri bir listede toplayacağız.


num_but_cat = [col for col in df.columns if (str(df[col].dtype) in ['int64', 'float64']) and (df[col].nunique() < 10)]
# Tipi integer ya da float olup aldığı farklı değerlerinin sayısı belirli bir
# sayının üzerinde olan değişkenleri belirleyik num_but_col isimli bir listede topladık.


## Şimdi kardinalitesi yüksek olan değişkenleri belirleyeceğiz. Bazı kategorik değişkenler veri setinden
## anlam çıkaramayacağımız kadar çok farklı çeşitte değere sahip olabilirdi. Bu durumda buradan anlamlı
## bir yorum çıkarmamız mümkün olmazdı.
## Örneğin değişkenlerimizden birisi yolcuların isimleri olsaydı burada muhtemelen gözlem sayısı kadar
## farklı değer olacaktı. Bu bir kategorik değişken olmasına rağmen istatistiksel açıdan bir
## anlam taşımamaktadır. Bunlara kardinalitesi yüksek değişkenler denir. Ölçüm değeri taşımayacak
## kadar fazla sınıfı vardır.

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and col in cat_cols]
# Kategorik değişkenler içerisinden aldığı farklı değer sayısı 20'den fazla olanları belirleyip bir liste içine yazdık

cat_cols = cat_cols + num_but_cat
# Kategorik olduğunu tesipt ettiğimiz bütün değişkenleri cat_cols listesinin içinde topladık.

cat_cols = [col for col in cat_cols if col not in cat_but_car]
# Kategorik olup kardinalitesi yüksek olan değişkenleri çıkarmak adına
# cat_cols listesine sadece kardinalitesi yüksek olmayanları bıraktık

df[cat_cols].nunique()  # seçtiğimiz değişkenlerin non-unique değerler sayısına


# bakarak metodumuzun doğrukluğunu kontrol ediyoruz.


## Şimdi bu yaptığımız analizlerin hepsini yaptıracağımız bir fonksiyon yazacağız:
## 1. kendisine verilen sınıfların value_countlarını yazsın
## 2. sınıfların yüzdelik bilgilerini yazdıralım

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),  #ilk sütuna col_name değişkenin aldığı değerlerin sayılarını yazar
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))  #ikinci sütuna Ratio isimini verir ve oranları yazar.
    print("#########################################")

# fonksiyona girdiğimiz dataframe'in içindeki col_name değişkeninin aldığı değer sayılarını
# ve bunların yüzdelerini ekrana yazdıran bir fonksiyon yazdık.

for col in cat_cols:  #Bu fonksiyonu bir döngüye koyarak bütün kategorik değişkenler için çıktı verir
    cat_summary(df, col)

# Şimdi bu fonksiyona bir de bu değişkenlerin grafiklerini gösterme özelliği ekleyelim

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################")

    if plot:  #fonskyiona bir de kategorik değişken grafiği gösterme komutu ekledik.
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

for col in cat_cols:
    cat_summary(df, col)

#yukarıdaki döngüyü çalıştırınca bir hatay ile karşılaştık.
#"adult_male" değişkeni 'bool' tipindedir ve countplot 'bool' tipindeki değişkenleri grafiğe dökmez!!!

for col in cat_cols:
    if df[col].dtypes == 'bool':
        print("This is not plotted since it is a variable of boolean type!")
    else:
        cat_summary(df, col, plot=True)

# Şimdi de 'bool' olan değişkenleri 'int'e dönüştürelim ve bu şekilde grafiği görüntüleyelim

for col in cat_cols:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)  #.astype() metodu ilgili veri türünü integera dönüştürür
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)


#################################################################
# 3. Sayısal Değişken Analizi - Analysis of Numerical Variables #
#################################################################

df[["age", "survived"]].describe().T  #age ve survived değişkenlerini describe ettirdik

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
#integer ve float veri tipindeli değişkenleri num_cols listesine ekledik
#fakat bunların içinde numerik olmasına rağmen kategorik olan değişkenler var

num_cols = [col for col in num_cols if col not in cat_cols]
#sadecee num_cols'da olup cat_cols'da olmayan değişkenleri listemizde bıraktık

# Şimdi verilen veri setinin verilen değişkenini describe edecek fonksiyonu yazıyoruz
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

# Şimdi bu işlemi bütün sayısal değişkenler için yapacak olan döngüyü yazıyoruz
for col in num_cols:
    print(f"The description of {col} is as follows:")
    num_summary(df, col)
    print("#############")

# Şimdi bu fonksyiona ilgili değişkenin grafiğini göstermesi özelliğini ekleyeceğiz

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

#Bu fonksiyonu bir döngü ile bütün sayısal değişkenlere uyguluyoruz
for col in num_cols:
    print(f"The description of {col} is as follows:")
    num_summary(df, col, plot=True)
    print("#############")


################################################################
# 4. Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi #
#        Capturing Variables and Generalizing Operations       #
################################################################


#Bu bölümde öyle bir fonksiyon yazacağız ki bize hem kategorik, hem sayısal
#hem de kardinal değişkenleri verecek

def grab_col_names(dataframe, cat_th=10, car_th=20): #bu parametrelerin ne anlama geldiğini aşağıdaki docstringde görebiliriz.
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen veri setidir.
    cat_th: int, float
        numerik fakat kategorik değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik degisken listesil
    num_cols: list
        Numerik degisken listesi
    cat_but_car: list
        Kategorik görinümlü kardinal degisken listesi

    Notes
    -------
        cat_cols + num_cols + cat_but_car = toplam degisken sayisi
        num_but_cat cat_cols'un içerisinde.

    """

    #kategorik ve kardinal değişkenleri oluşturacak bölüm
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtype) in ['object', 'bool', 'category']]

    num_but_cat = [col for col in dataframe.columns if (str(dataframe[col].dtype) in ['int64', 'float64']) and (df[col].nunique() < cat_th)]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and col in cat_cols]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #sayısal değişkenleri oluşturacak bölüm
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    #Raporlama amacıyla bir kaç print ifadesi ekleyebiliriz
    print(f"Observations: {dataframe.shape[0]}")  #gözlem sayısını verir
    print(f"Variables: {dataframe.shape[1]}")  #değişken sayısı
    print(f'cat_cols: {len(cat_cols)}')  #kategorik değişkenlerin sayısı
    print(f'num_cols: {len(num_cols)}')  #numerik değişkenlerin sayısı
    print(f'cat_but_car: {len(cat_but_car)}')  #kategorik fakat kardinal olan değişkenlerin sayısı
    print(f'num_but_cat: {len(num_but_cat)}')  #sayısal olmasına rağmen kategorik sayılan değişkenlerin sayısı

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)  #fonksiyonu çalıştırıyoruz.

# Şimdi daha önce tanımladığımız kategorik ve numerik değişkenleri betimleyen fonksiyonları bir araya getirelim

#Kategorik değişkenleri betimleyen fonksiyon
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################")

#Yukarıdaki fonksiyonu bütün kategorik değişkenlere uygulayan döngü
for col in cat_cols:
    cat_summary(df, col)

#Sayısal değişkenleri betimleyip grafikleyen fonksiyon
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

#Yukarıdaki fonksiyonu bütün sayısal değişkenlere uygulayan döngü
for col in num_cols:
    print(f"The description of {col} is as follows:")
    num_summary(df, col, plot=True)
    print("#############")



#############
### BONUS ###
#############


# Burada 'bool' tipindeki kategorik değişkenleri grafiğe dökebilmek için tip değiştirme işlemi yapacağız

df = sns.load_dataset("titanic")
for col in df.columns:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)  #eğer değişkenin tipi 'bool' ise bunu 'int' yapıyoruz


#Değişkenleri gruplandıralım
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Kategorik değişkenleri hem tanımlayıp hem de grafiğe döken fonksiyon
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################")

    if plot:  #fonskyiona bir de kategorik değişken grafiği gösterme komutu ekledik.
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

#Yukarıdaki fonksiyonu bütün kategorik değişkenlere uygulayan döngü
for col in cat_cols:
    cat_summary(df, col, plot=True)


###########################################################
# 4. Hedef Değişken Analizi - Analysis of Target Variable #
###########################################################

#burada "survived" değişkenini diğer değişkenlere göre değerlendireceğiz
def grab_col_names(dataframe, cat_th=10, car_th=20): #bu parametrelerin ne anlama geldiğini aşağıdaki docstringde görebiliriz.
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen veri setidir.
    cat_th: int, float
        numerik fakat kategorik değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik degisken listesil
    num_cols: list
        Numerik degisken listesi
    cat_but_car: list
        Kategorik görinümlü kardinal degisken listesi

    Notes
    -------
        cat_cols + num_cols + cat_but_car = toplam degisken sayisi
        num_but_cat cat_cols'un içerisinde.

    """

    #kategorik ve kardinal değişkenleri oluşturacak bölüm
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtype) in ['object', 'bool', 'category']]

    num_but_cat = [col for col in dataframe.columns if (str(dataframe[col].dtype) in ['int64', 'float64']) and (df[col].nunique() < cat_th)]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and col in cat_cols]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #sayısal değişkenleri oluşturacak bölüm
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    #Raporlama amacıyla bir kaç print ifadesi ekleyebiliriz
    print(f"Observations: {dataframe.shape[0]}")  #gözlem sayısını verir
    print(f"Variables: {dataframe.shape[1]}")  #değişken sayısı
    print(f'cat_cols: {len(cat_cols)}')  #kategorik değişkenlerin sayısı
    print(f'num_cols: {len(num_cols)}')  #numerik değişkenlerin sayısı
    print(f'cat_but_car: {len(cat_but_car)}')  #kategorik fakat kardinal olan değişkenlerin sayısı
    print(f'num_but_cat: {len(num_but_cat)}')  #sayısal olmasına rağmen kategorik sayılan değişkenlerin sayısı

    return cat_cols, num_cols, cat_but_car

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################")

    if plot:  #fonskyiona bir de kategorik değişken grafiği gösterme komutu ekledik.
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_summary(df, "survived")


### Hedef Değişkenin Kategorik Değişkenlerle Analizi ###


df.groupby("sex")["survived"].mean()  #survived değişkeninin sex değişkeninin aldığı değerlere göre ortalaması
                                        # yani survived değişkeninde female ve male olanların ortalaması...

# Şimdi bu işlemi bir fonksiyona çevirelim
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))
#Bu fonksiyon target değişkeninin categorical_col değişkeninde aldığı değerlerin ortalamasını hesaplar

target_summary_with_cat(df, "survived", "sex")

# Şimdi bu fonksiyonu her kategorik değişkende dönecek döngüyü yazıyoruz
for col in cat_cols:
    target_summary_with_cat(df, "survived", col)


### Hedef Değişkenin Sayısal Değişkenlerle Analizi ###

df.groupby("survived")["age"].mean()  #survived değişkeninin aldığı değerlerin yaş ortalamalarını verir

df.groupby("survived").agg({"age":"mean"})  #aynı işlemi aggregation kullanarak bu şekilde yapabiliriz

# Şimdi bu işlemi her sayısal değişkenler için yapabilecek fonksiyonu yazalım
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}))

# Bu işlemi bütün numerik değişkenlerde yapmak için gereken döngü
for col in num_cols:
    target_summary_with_num(df, "survived", col)


##################################################
# 5. Korelasyon Analizi - Analysis of Corelation #
##################################################


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 2:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtypes in [int, float]]
#numerik değişkenleri bir listeye ekledik

corr = df.corr()  #bütün değişkenlerin birbiri ile korelasyonunu hesaplar

## Çalışmalarda yüksek korelasyonlu değişken ikililerinin bir arada bulunmamasını isteriz.
## Çünkü ikisi de aynı bilgiyi taşırlar. O yüzden yüksek korelasyonlu değişkenlerden birini veriden çıkarmalıyız.

## yüksek korelasyonlu değişkenleri gözlemlemek için aşağıdaki ısı haritasını kullanabilriz
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu" )
plt.show()


### Yüksek Korelasyonlu Değişkenlerin Silinmesi ###


cor_matrix = df[num_cols].corr().abs()  #veri setindeki değişkenlerin korelasyonlarının mutlak değereini bir matrixe koyduk.
                                #bu matrixin diagonal entryleri 1'dir ve simetrik bir matrixtir.

#Şimdi bu tekrar eden verileri ve diagonaldeki 1'leri sileceğiz
upper_tri_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
#bunu yapmak için önce bir numpy arrayi oluşturduk elimizdeki matrix ile aynı ölçüde: np.ones(cor_matrix.shape)
#bunu 1'lerle dolduruduk: .astype(bool)
#bu 1'leri upper triangular olacak şekilde yerleştirdik: np.triu(...)
#orjinal matrixin yalnızca burada 1 bulunan hücrelerindeki değerleri aldık: .where(...)

drop_list = [col for col in upper_tri_matrix.columns if any(upper_tri_matrix[col] > 0.90)]
#korelasyonu 0.90'dan fazla olan değişkenleri belirledik ve drop_list isimli bir listede kaydettik

df.drop(drop_list, axis=1)

# Şimdi bu işlemi bir fonksiyona dönüştürelim

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()  #veri setinin korelasyonunun hesapladı
    cor_matrix = corr.abs()  #bu değerlerin mutlak değerlerini aldı
    upper_triangular_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
                                #diagonal ve altındaki entryleri sildi
    drop_list = [col for col in upper_triangular_matrix.columns if any(upper_tri_matrix[col] > corr_th)]
                                #yüksek korelasyonlu değişkenleri bir listede topladı

    if plot:  #korelasyonların ısı haritasını oluşturdu
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list  #yüksek korelasyonlu değişkenlerin listesini return etti

#Şimdi yukarıdaki fonksiyonu kullanarak yüksek korelasyonlu değişkenlerden kurtularak tekrar ısı haritası oluşturalım

drop_list = high_correlated_cols(df)  #silmek istediğimiz elemanları yukarıdaki fonksiyonu kullanarak bir listeye aldık
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)
#orjinal veri setinden df.drop(drop_list, axis=1) komutuyla istenmeyen değişkenleri silip o şekilde grafiği çizdirdik
