###################################################
# Sorting Products
###################################################

###################################################
# Uygulama: Kurs Sıralama
###################################################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/product_sorting.csv")
print(df.shape)
df.head(10)

####################
# Sorting by Rating
####################

df.sort_values("rating", ascending=False).head(20)
# bu şekilde sıraladdığımızda satın alma sayısı ve yorum yapma sayısı faktörlerini gözden kaçırmış oluyoruz
# bunlar "sosyal ispat" presibi için hesaba katılmak zorundadır.

####################
# Sorting by Comment Count or Purchase Count
####################

df.sort_values("purchase_count", ascending=False).head(20)
df.sort_values("commment_count", ascending=False).head(20)

####################
# Sorting by Rating, Comment and Purchase
####################

# burada akla ilk gelecek şey bütün bu değişkenleri çarparak bir metrik elde etmek olabilir.
# fakat bu durum yorum sayısı çok olanlar az olanlara kıyasla fazla avantajlı konuma koyabilir
# benzer şekilde satın alma sayısı örneğin 4 basamaklı olan bir eğitim 3 basamaklı olan bir
# eğitimi kolayca maskeleyecektir.
# bunu düzenlemek için bir çeşit normalizasyon işlemi yapacağız
# değerlerin hepsini 1-5 arasına scale edeceğiz

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df["commment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

# bu aşamda dilersek bu skorların hepsini çarparak her bir ürün için betimleyici bir metrik elde etmiş oluruz
# fakat bu skorların etkisi birbirleri ile aynı olmayabilir. Bu sebeple bunların ağırlıklı ortalamalarını alabiliriz

df["weighted_sorting_score"] = (df["commment_count_scaled"] * 32/100 + \
    df["purchase_count_scaled"] * 26/100 + \
    df["rating"] * 42/100)

df.sort_values("final_score", ascending=False)

# bu süreci fonksiyonlaştırabiliriz
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["commment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)

df["weighted_sorting_score"] = weighted_sorting_score(df)



####################
# Bayesian Average Rating Score
####################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

# Sadece ratinglere bakarak anlamlı yorumlar yapmak mümkün müdür?
# Puanların dağılımını ağırlıklı olarak hesaba katarsak mümkün olabilir...

def bayesian_average_rating(n, confidence=0.95):  # n=her bir ratingden kaç tane olduğunun bilgisi
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)
df.sort_values("bar_score", ascending=False)

df[["rating", "bar_score"]].describe().T
df["bar_score"].describe().T

### SORU: bar_score'un standart sapması neden daha yüksek?



####################
# Hybrid Sorting: BAR Score + Diğer Faktorler
####################

# SUMMARY:
# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Other Factors (New)

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score * bar_w/100 + wss_score * wss_w/100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)




############################################
# Uygulama: IMDB Movie Scoring & Sorting
############################################

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/movies_metadata.csv",
                 low_memory=False)  # DtypeWarning kapamak icin

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

########################
# Vote Average'a Göre Sıralama
########################

df.sort_values("vote_average", ascending=False).head(20)


df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)

from sklearn.preprocessing import MinMaxScaler

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])



########################
# vote_average * vote_count
########################

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)


########################
# IMDB Weighted Rating
########################

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85

M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df.sort_values("average_count_score", ascending=False).head(20)

# Deadpool filmi için bu hesaplamayı yapalım:
weighted_rating(7.40000, 11444.00000, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)
df.head(20)



####################
# Bayesian Average Rating Score
####################

# the first 5 in our list so far
# 12481                                    The Dark Knight
# 314                             The Shawshank Redemption
# 2843                                          Fight Club
# 15480                                          Inception
# 292                                         Pulp Fiction


def bayesian_average_rating(n, confidence=0.95):  # n=her bir ratingden kaç tane olduğunun bilgisi
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

# Esaretin Bedeli filmi için
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

# Baba filmi için
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)

df.sort_values("bar_score", ascending=False).head(20)


# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.
#
# See also the complete FAQ for IMDb ratings.