############################################################
#                   PYTHON ALIŞTIRMALARI                   #
############################################################

# Görev 1: Verilen değerlerin veri yapılarını inceleyiniz. #
x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23 < 22
l = [1, 2, 3, 4]
d = { "Name": "Jake",
      "Age": 27,
      "Address": "Downtown"}
t = ("Machine Learning", "Data Science")
s = {"Python", "Machine Learning", "Data Science", "Python"}

# Çözüm1: Verilen nesnelerin veri yapılarını sorgulamak için type() fonksiyonunu kullanabiliriz.

list = [x, y, z, a, b, c, l, d, t, s] #Değişkenlerin isimlerini bir listede topladık

for var in list: # "list" adlı  listenin her elemanı üzerinde döngü yapar. Her elemanı "var" adlı geçici bir değişkene atar.
    var_name = [name for name, value in locals().items() if value is var][0]
        # Yukarıdaki satır, her döngü adımında, locals() fonksiyonunu kullanarak mevcut yerel değişkenlerin
        # adlarını ve değerlerini içeren bir sözlük elde eder.
        # Daha sonra, bu sözlükteki her öğeyi (name ve value) kontrol eder ve value değişkeni var ile eşleştiğinde,
        # bu değişkenin adını var_name değişkenine atar.
        # Bu işlemi bir liste anlamında kullanmaktan dolayı [0] ile listenin ilk (ve tek) elemanını seçer.
        # Bu sayede var_name, var değişkeninin adını içerir.
    print(f"{var_name} degiskeninin veri Tipi: {type(var)}")
        # Bu satır, formatlı bir string kullanarak var_name değişkeninin adını ve
        # var değişkeninin veri tipini ekrana yazdırır. type(var) ifadesi,
        # var değişkeninin veri tipini elde etmek için kullanılır.

# GÖrev 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız. #

text = "The goal is to turn data into information, and information into insight"

# Çözüm2: upper(), replace() ve split() metotlarını kullanabiliriz. #

text.upper().replace(",", " ").replace(".", " ").split()
# upper() metodu verilen stringin bütün harflerini büyük harfe çevirir.
# replace("x", "y") stringdeki "x" harflerini "y" harfiyle değiştirir.
# split() metodu stringi boşluklara göre parçalara ayırır.
# split("x") stringi "x" karakterine göre parçalara ayırır.

# Görev 3: Verilen listeye aşağıdakia dımları uygulayınız.#

lst = ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'N', 'C', 'E']

# Çözüm 3:

# Adım1: Verilen listenin eleman sayısına bakınız.
len(lst)

# Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
lst[0]
lst[10]

# Adım3: Verilen liste üzerinden["D", "A", "T", "A"] listesi oluşturunuz.
lst[0:4]

# Adım4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)

# Adım5: Yeni bir eleman ekleyiniz.
lst.append("!")

# Adım6: Sekizinci indekse"N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")

# Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız. #

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

# Çözüm 4: #

# Adım1: Key değerlerine erişiniz.
dict.keys()
# Adım2: Value'laraerişiniz.
dict.values()
# Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Daisy"][1] = 13
# Adım4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict["Ahmet"] = ["Turkey", 24]
# Adım5: Antonio'yu dictionary'den siliniz.
dict.pop("Antonio")


# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları
# ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.  #

l = [2, 13, 18, 93, 22]

# Çözüm 5 #
def func(list): # parametre olarak içine  "list" nesnesini alacak "func" isminde bir fonksiyon tanımlıyoruz.
    odd = [] # "odd" isimli boş bir liste tanımlıyoruz
    even = []  # "even" isimli boş bir liste tanımlıyoruz

    for i in list: #list içerisindeki elemanlar üzerinde dönen ve i isimli bir değişkene atayan bir döngü oluşturuyoruz.
        if i % 2 == 0: # eğer i'nin 2'ye bölümünden kalanı 0 ise (yani i çift sayı ise) bu koşul sağlanır
            even.append(i) # i değerini "even" isimli listeye ekliyoruz.
        else: # eğer yukarıdaki koşul sağlanmıyorsa (yani i çift değilse yani i tek ise) bu koşul gerçekleşir
            odd.append(i) # i değerini "odd" isimli listeye ekliyoruz.
    return even, odd # oluşturulan listeleri return ediyoruz.

even_list, odd_list = func(l) # "func" fonksiyonun return ettiği listeleri "even_list" ve "odd_list" isimli iki değişene atıyoruz.

# Görev 6: Görev 6:Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi
# öğrenci sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız. #

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

# Çözüm 6: #

for index, student in enumerate(ogrenciler): # enumerate kullanarak hem "ogrenciler" listesindeki elemanların üzerinde hem de
                                             # bunların indexinde dönecek bir döngü oluşturduk
    if index < 3: # eğer index 3'ten küçükse bu koşul gerçekleşir
        print("Mühendislik Fakültesi " + str(index + 1) + ". öğrenci: " + student) # str() fonksiyonuyla index değişkeninin integer olan
                                                                                   # integer olan türünü string türüne dönüştürüyoruz.
    else: # yukarıdaki koşul sağlanmazsa bu satırın altıdaki satır uygulanır
        print("Tıp Fakültesi " + str(index - 2) + ". öğrenci: " + student)

# Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve
# kontenjan bilgileri yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.

ders_kodu = ["CMPE105", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

# Çözüm 7 #

for ders, puan, kont in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {puan} olan {ders} kodlu dersin kontenjanı {kont} kişidir")

# Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1.küme 2.kümeyi kapsiyor ise ortak elemanlarını
# eğer kapsamıyor ise 2.kümenin 1.kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir. #

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

# Çözüm 8: #
def my_func(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

my_func(kume1, kume2)
