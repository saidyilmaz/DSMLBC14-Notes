
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import datetime as dt
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


### Adım 1: Veri setlerini okutunuz.

df_att = pd.read_csv("Week 9 - Machine Learning III/Scoutium-220805-075951/scoutium_attributes.csv", sep=";")
df_pot_labels = pd.read_csv("Week 9 - Machine Learning III/Scoutium-220805-075951/scoutium_potential_labels.csv", sep=";")

### Adım 2: Veri setlerini "task_response_id", 'match_id', 'evaluator_id', "player_id" değişkenleri üzerinden birleştiriniz
df_ = pd.merge(df_att, df_pot_labels, how="left", on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])

df = df_.copy()


### Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df = df[~(df["position_id"] == 1)]

### Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.

df = df[~(df["potential_label"] == "below_average")]

### Adım 5: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların
# oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz. Daha sonra “reset_index”
# fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini
# stringe çeviriniz.

df_p = pd.pivot_table(df,
                          values="attribute_value",
                          index=["player_id", "position_id", "potential_label"],
                          columns="attribute_id")

df_p.reset_index(drop=False ,inplace=True)

df_p.dtypes

df_p["player_id"] = df_p["player_id"].astype(str)
df_p["position_id"] = df_p["position_id"].astype(str)


### Adım 6: LabelEncoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal
# olarak ifadeediniz.

df_p["potential_label"].unique()
def label_encoder(dataframe, binary_col):
    """
    Fonksiyon verilen veri setindeki ilgili değişkenleri label encoding sürecine tabii tutar.

    Parameters
    ----------
    dataframe: Veri setini ifade eder.
    binary_col: Encode edilecek olan değişkenleri ifade eder

    Returns
    -------
    Encoding işlemi yapılmiş bir şekilde "dataframe"i return eder

    Notes
    -------
    Fonksiyonun "from sklearn.preprocessing import LabelEncoder" paketine bağımlılığı bulunmaktadır.

    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(df_p, "potential_label")

### Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.

num_cols = df_p.columns[3:]

### Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

ss = StandardScaler()
df_p[num_cols] = ss.fit_transform(df_p[num_cols])

### Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine
# öğrenmesi modeli geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

X = df_p[num_cols]

y = df_p["potential_label"]

models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ("LightGBM", LGBMClassifier(verbose=-1))]

for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))


# LGBM Modelini seçerek hiperparametre optimizasyonu yapıyorum:

lgbm_model = LGBMClassifier()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=-1).fit(X, y)

# cv=3 iken {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 500}
# cv=5 iken {'colsample_bytree': 1, 'learning_rate': 0.01, 'n_estimators': 500}
# cv=10 iken {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 1500}

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(final_model,
                            X, y,
                            cv=3,
                            scoring=["roc_auc", "f1", "precision", "recall", "accuracy"])

cv_results['test_roc_auc'].mean()
cv_results['test_f1'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results['test_accuracy'].mean()

### Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin
# sıralamasını çizdiriniz.
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)
