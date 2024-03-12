################################
# Telco Customer Churn Project #
################################

###############
# İş Problemi:
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri
# sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını,
# kaldığını veya hizmete kaydolduğunu gösterir.
###############

##############
# Değişkenler:
##############

# CustomerId
# Müşteri İd’si
# Gender
# Cinsiyet
# SeniorCitizen
# Müşterinin yaşlı olup olmadığı (1, 0)
# Partner
# Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents
# Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure
# Müşterinin şirkette kaldığı ay sayısı
# PhoneService
# Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines
# Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService
# Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity
# Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup
# Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection
# Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport
# Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV
# Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies
# Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract
# Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling
# Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod
# Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges
# Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges
# Müşteriden tahsil edilen toplam tutar
# Churn
# Müşterinin kullanıp kullanmadığı (Evet veya Hayır)


#####################
# Görev 0: Gerekli import ve ayarlamalar
#####################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df_ = pd.read_csv("Week 7-8-9 - Machine Learning/TelcoChurn/Telco-Customer-Churn.csv")
df = df_.copy()

#####################################
### GÖREV 1: KEŞİFÇİ VERİ ANALİZİ ###
#####################################

# Adım 0: Veriye ön bakış
df.shape
df.dtypes
df.isnull().sum()
df.describe().T

# Not: TotalCharges object tipinde fakat numeric olmalı!!!!



# Adım 1: Numerik ve kategorik değişkenleri yakalayınız
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Adım 2: Gerekli düzenlemeler (type düzeltmesi gibi)
# Not: TotalCharges object tipinde fakat numeric olmalı!!!!
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Churn değişkeni 1 ve 0'lardan oluşmalı!
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Değişkenleri yeniden kategorize edelim
cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımı
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col,plot=True)


def cat_summary(dataframe, col_name, plot=False):
    """

    Fonksiyon, veri setinde yer alan kategorik, numerik vs... şeklinde gruplandırılan değişkenler için özet bir çıktı
    sunar.

    Parameters
    ----------
    dataframe : Veri setini ifade
    col_name : Değişken grubunu ifade eder
    plot : Çıktı olarak bir grafik istenip, istenmediğini ifade eder, defaul olarak "False" gelir

    Returns
    -------
    Herhangi bir değer return etmez

    Notes
    -------
    Fonksiyonun pandas, seaborn ve matplotlib kütüphanelerine bağımlılığı vardır.

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=dataframe, x=col_name, palette="Set3")
        plt.title(f'{col_name} Değişkeninin Dağılımı')
        plt.xticks(rotation=45)
        # Her sütunun üzerine frekansları yazdırma
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2., p.get_height(), f'\n{p.get_height()}',
                    ha='center', va='bottom', fontsize=8, color='black')
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)



# Adım 4: Hedef değişken ile kategorik değişken analizi yapınız:

for col in cat_cols:
    print(df.groupby("Churn").agg({col: "value_counts"}))



# Adım 5: Aykırı değer gözlemi yapınız:

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Bir dataframe için verilen ilgili kolondaki aykırı değerleri tespit edebilmek adına üst ve alt limitleri belirlemeyi
    sağlayan fonksiyondur

    Parameters
    ----------
    dataframe: "Dataframe"i ifade eder.
    col_name: Değişkeni ifade eder.
    q1: Veri setinde yer alan birinci çeyreği ifade eder.
    q3: Veri setinde yer alan üçüncü çeyreği ifade eder.

    Returns
    -------
    low_limit, ve up_limit değerlerini return eder
    Notes
    -------
    low, up = outlier_tresholds(df, col_name) şeklinde kullanılır.
    q1 ve q3 ifadeleri yoru açıktır. Aykırı değerle 0.01 ve 0.99 değerleriyle de tespit edilebilir.

    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
    Bir dataframein verilen değişkininde aykırı gözlerimerin bulunup bulunmadığını tespit etmeye yardımcı olan
    fonksiyondur.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(f'{col}: {check_outlier(df,col)}')


### Note to self: Not sure how to implement this part yet or if this is necessary at all....
def rare_analyser(dataframe, target, cat_cols):
    """
    Verilen veri setindeki hedef değişkene göre değişken grubundaki nadir gözlemleri analiz eder
    Parameters
    ----------
    dataframe : Veri setini ifade eder.
    target : Hedef değişkeni ifade eder.
    cat_cols : Değişken grubunu ifade eder

    Returns
    -------
    Herhangi bir değer retrun etmez.
    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Churn", cat_cols)



# Adım 6: Eksik değer gözlemi yapınız
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    """

    Bir veri setindeki eksik gözlemleri tespit etmek için kullanılan fonksiyondur. Fonksiyon kullanıcıya "n_miss" ile
    eksik gözlem sayısını "ratio" ile de eksik gözlemlerin değişkende kapladığı yeri yüzdelik olarak ifade eder

    Parameters
    ----------
    dataframe: Veri setini ifade eder
    na_name: Eksik gözlem barındıran değişkenleri ifade eder

    Returns
    -------
    Eğer na_name parametleri True olarak girildiyse eksik gözlem barındıran değişkenleri liste olarak return eder

    Notes
    -------
    Fonksiyonun numpy ve pandas kütüphanelerine bağımlılığı vardır.

    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)



#####################################
### GÖREV 2: ÖZELLİK MÜHENDİSLİĞİ ###
#####################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

df.dropna(inplace=True)
# eksik değerler toplam verinin yüzde 0.1'ini oluşturduğu için silmeyi tercih ettim



# Adım 2: Yeni değişkenler oluşturunuz.

# Streaming servislerinden yararlanıp yararlanmadığını belirten değişken
df['TotalStreamingServices'] = (df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')

# Online servis değişkeni
df["TotalOnlineServices"] = (df['OnlineSecurity'] == 'Yes') | (df['OnlineBackup'] == 'Yes')

# Ödeme metotları ile ilgili bilgileri bir araya getiren değişken
df['CombinedPayment'] = df['PaymentMethod'] + '_' + df['PaperlessBilling']

# Senior ve gender değişkenlerinin birleşimi
df["GenderSeniorCombined"] = df['gender'] + '_' + df['SeniorCitizen'].astype(str)

# Alınan servis sayısı
df['ServiceCount'] = df[
    ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
    lambda row: row == 'Yes', axis=1).sum(axis=1)

# Hizmet başına ödenen miktarı hesaplama
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for service in services:
    df[f'{service}_CostPerService'] = df['TotalCharges'] * (df[service] == 'Yes')

# Sözleşme Süresi
contract_duration = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df['ContractDuration'] = df['Contract'].map(contract_duration)

# Müşteri sadakati
df['LoyaltyScore'] = df['tenure'].apply(lambda x: x // 12)  # Her 12 ay için 1 puan

# tenure segmentleri
df["new_tenure_cat"] = pd.qcut(df["tenure"], q=3, labels=["short_term", "mid_term", "long_term"])



# Adım 3: Encoding işlemlerini yapınız:
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Binary değişkenlerin encode edilmesi
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Adım 4: Numerik değişkenler için standartlaştırma işlemi
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.describe()
df[num_cols].describe().T



### Görev 4: Modelleme

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyiniz

# a) Logistic Regression Modelini Kurma
y = df["Churn"]

X = df.drop(["Churn", "customerID"], axis=1)

log_model = LogisticRegression(max_iter=1000).fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

# Modelin başarısını confusion matrix ile belirleme
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# Modelin başarısını ROC curve ile belirleme
y_prob = log_model.predict_proba(X)[:, 1]

roc_auc_score(y, y_prob)

# Accuracy: 0.81
# Precision: 0.68
# Recall: 0.54
# F1: 0.60
# Auc: 0.85

# Modelin geçerliliğini Holdout yöntemi ile kontrol etmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

roc_auc_score(y_test, y_prob)

# Accuracy: 0.80
# Precision: 0.64
# Recall: 0.52
# F1: 0.57
# Auc: 0.83

# Modelin geçerliliğini 10-fold cross validation yöntemi ile kontrol etmek

log_model = LogisticRegression(max_iter=500).fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.80

cv_results['test_precision'].mean()
# Precision: 0.66

cv_results['test_recall'].mean()
# Recall: 0.53

cv_results['test_f1'].mean()
# F1-score: 0.58

cv_results['test_roc_auc'].mean()
# AUC: 0.84

plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# b) KNN Modeli Kurma
knn_model = KNeighborsClassifier().fit(X, y)

y_pred = knn_model.predict(X)
y_prob = knn_model.predict_proba(X)[:, 1]


print(classification_report(y, y_pred))
roc_auc_score(y, y_prob)
# Accuracy: 0.84
# Precision: 0.72
# Recall: 0.65
# F1: 0.69
# Auc: 0.89

# 10-fold cross validation
cv_results = cross_validate(knn_model, X, y, cv=10, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.77

cv_results['test_precision'].mean()
# Precision: 0.57

cv_results['test_recall'].mean()
# Recall: 0.53

cv_results['test_f1'].mean()
# F1-score: 0.55

cv_results['test_roc_auc'].mean()
# AUC: 0.77



# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz
# hiperparametreler ile modeli tekrar kurunuz.
knn_model = KNeighborsClassifier().fit(X, y)
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.79

cv_results['test_precision'].mean()
# Precision: 0.63

cv_results['test_recall'].mean()
# Recall: 0.53

cv_results['test_f1'].mean()
# F1-score: 0.58

cv_results['test_roc_auc'].mean()
# AUC: 0.82