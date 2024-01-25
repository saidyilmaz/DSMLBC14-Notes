#################
# VERİ YAPILARI #
#################

# type() fonksiyonu içerisine girilen değişkenin türünü verir

a = 3
type(a) # a'nın veri tipi 'int' yani integer'dır.
x = 46.3
type(x) # x'in veri türü 'float'tur.

y = 2j + 1
type(y) # y'nin veri tipi 'complex'tir.

z = "hello ai era"
type(z) # z'nin veri tipi 'str' yani string'dir.

type(True)         # bu veri tipleri 'bool'dur
type(False)
type(True != False)

type(3 == 2)

list = ["TL", "$", "£"]
type(list) # bu veri tipi 'list'tir.

x = {"name": "Peter", "Age": 36}
type(x) # bu veri tipi 'dict'tir. dict'ler bir key ve value değerlerinden oluşurlar.

x = ("t1", "t2", "t3")
type(x) # bu veri tipi 'tuple'dır.

x = {"abc", "def", "ghj"}
type(x) # bu veri tipi 'set'tir.

# Sayılarla işlemler yapabiliriz
a = 5 # a değişkenine 5 değerini atarız, böylece 'int' tipinde bir değişken olur
b = 10.5 # b değişkenine 10.5 değerini atarız, böylece 'float' tipinde bir değişken olur
a * 5 # a değişkeninin aldığı değerin 5 katını gösterir. a değişkeninin değerini değiştirmez!
c = a / 3 # c değşkenine a değişkenini aldığı değerin 3'e bölümünü atar.

int(a * b / 10) # a*b/10 değeri bir float olmasına rağmen bunu int' çevirir ve virgülden sonrasını atar.

name2 = """ alkfg dglkjag jdfkljam 
akj hdfşadshf jaf """ # stringleri ekrana sığdırmak için iki satıra bölebiliriz bunun için üç tane " işareti kullanmalıyız.

name = "John"
name[1:4] # stringlerin içerdiği harflerin belirli bir kısmına ulaşmak için string[] formatını kullanabiliriz
          # string[a:b] bize stringin a indexinden başlayarak b indexine kadar olan (b indexi dahil değil) kısmını verir

"jaf" in name2 # name2 stringinde "jaf" ifadesi varsa True yoksa False boolian değerlerini verir

### String Metotları ###

dir(str) # str tipindeki verilere uygulayabilceğimiz metotların listesini verir.

len("abcdef") # len() fonksiyonu stringin uzunluğunu veren bir fonksiyondur

"miuul".upper() # .upper() metodu verilen stringin bütün harflerini büyük harf yapan metottur.

hi = "hello ai era"
hi.replace("l", "p") # verilen stringdeki "l" harflerinin hepsini "p" harfiyle değiştirir.

"hello ai era".split("a") # verilen stringi "a" harflerinden bölerek alt stringler olşturur ve bunları bir listeye koyar.

"ofofofofoasdasdgoogoffadsa".strip()

"foo asdas".capitalize() # verilen stringin baş harfini büyük yapar

### Listeler ###

notes = [1, 2, 3, 4] #listeler sıralı, değiştirilebilirler ve kapsayıcıdırlar (yani birden fazla veri türünü saklayabilirler)
type(notes)

names = ["a", "b", 5, True, [1, 2, 3]] # birden fazla veri türünü barındıran bir liste.

names[0] = 3 # listelerin elemanlarına list[index] formatını kullanarak erişebiliriz.

names[1:4] # 1'inci indexten başlayarak 4'üncü indexe kadar (4 hariç) listenin elemanlarını barındıran bir liste verir.

names.append(100) # listenin sonuna "100" elemanını ekler

names.pop(1) # 1'inci indexteki elemanı yok eder. bu diğer elemanların indexini değiştirir.

names.insert(2, False)# verilen indexe verilen elemanı ekler.

### Sözlükler - Dictionaries ###

# sözlükler key:value şeklinde ikililer barındırırlar

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}

dictionary.keys() # key değerlerinin bulunduğu bir liste verir

dictionary.values() # value değerlerinin bulunduğu bir liste verir

dictionary["REG"] # dict[key] formatıyla key değerinin sahip olduğu value okunabilir.

"REG" in dictionary # "REG" key'inin dictionary içinde olup olmadığını bir boolean ile söyler

dictionary["REG"] = [1, "adas", 3] # "REG" key'inin sahip olduğu value değerini değiştirir ve onu [1, "adas", 3] yapar.
dictionary.items() # dictionary içindeki key:value ikililerini bir tuple haline getiriri ve bir liste içine koyar.