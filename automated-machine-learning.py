import streamlit as st
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import base64
import seaborn as sns
import missingno as msno
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing  import RobustScaler
import scipy.stats as stat
from xgboost import XGBClassifier
import lightgbm as lgb
import re
st.title("Create Your Own Model and Generate Code")
st.image('./automated-machine-learning.png')
st.write("""
## With this app, you can create and test your own model with your own data. 
## Currently only works for Classification Models.
### Let's start ! 
Load your own dataset or select dataset 
""")



uploaded_files = st.file_uploader("Upload new dataset CSV ", type="csv", accept_multiple_files=False)

try:
    df = pd.read_csv(uploaded_files)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    target_column = st.selectbox("Select your target Column", df.columns)
except:
    dataset_name = st.selectbox("Select Datasets", ("Titanic","Churn"))
    st.write(dataset_name)



    def get_dataset(dataset_name):
        if dataset_name == "Titanic":
            df = pd.read_csv("train.csv")
            df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
            target_column = "Survived"
        else:
            df = pd.read_csv("trainchurn.csv")
            df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
            target_column="churn"

        return df,target_column



    df,target_column = get_dataset(dataset_name)
if st.button("Generate Codes for Libraries"):
    st.code(f"""
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import missingno as msno
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing  import RobustScaler
import scipy.stats as stat """)
df_test = df
if st.button("Generate Codes for reading file"):
    st.code(f"""df = pd.read_csv("data.csv")
target_columns = "{target_column}"
""")
st.cache()
drop_true = st.selectbox("Drop columns that you want to drop (ID Columns etc.)",("Keep All Columns","Select  Columns to drop"))
if drop_true == "Select  Columns to drop":
    d_list = st.multiselect("Select Columns that will be dropped",df.columns)
    df = df.drop(d_list,axis=1)
    if st.button("Generate Codes for Columns to drop"):
        if drop_true == "Select  Columns to drop":
            st.code(f"""df = df.drop({d_list},axis=1)""")


visualization=st.sidebar.selectbox("Select an option for EDA visualization",("Do not show visualization","Show visualization"))
if visualization=="Show visualization":
    st.write(""" ### EDA """)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.pairplot(df, hue=target_column)
    st.pyplot()
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(),
                vmin=-1,
                cmap='coolwarm',
                annot=True);
    st.pyplot(fig)
    fig, ax = plt.subplots()
    df.isnull().sum().plot(kind='bar')
    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    msno.matrix(df)
    st.pyplot()
    if st.button("Generate Codes for visualization"):
        if visualization == "Show visualization":
            st.code(f"""
#Visialization
sns.pairplot(df, hue={target_column})
fig, ax = plt.subplots()
sns.heatmap(df.corr(),
              vmin=-1,
              cmap='coolwarm',
              annot=True);
df.isnull().sum().plot(kind='bar')
msno.matrix(df)""")
st.cache()


st.write(30*"--")
st.write("""
## Correlation (Data Cleaning)
""")
# find and remove correlated features
threshold = st.selectbox("Select threshold for find and remove correlated features", (1,0.9,0.8,0.7,0.6,0.5))
corCol=[]
numeric_cols = df.select_dtypes(exclude=["object"]).columns
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                corCol.append(colname)
    return col_corr
correlation(df,threshold=threshold)
df.drop(corCol, axis=1, inplace=True)
if len(corCol)>0:
    st.write(f"{corCol} columns correlations more than {threshold} and we dropped")
else:
    st.write("No columns exceeding the threshold you set for correlation")
if st.button("Generate Codes for Correlation"):
    st.code(f"""#find and remove correlated features
corCol=[]
numeric_cols = df.select_dtypes(exclude=["object"]).columns
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                corCol.append(colname)
    return col_corr
correlation(df,threshold={threshold})
df.drop(corCol, axis=1, inplace=True)""")


st.write(30*"--")
st.write("""
## Missing Imputation
""")
threshold_na = st.selectbox("Select threshold for the percentage of missingness and remove these features", (100, 90, 80, 70, 60, 50,40,30,10))
drop_list = []
def missing_drop(df,treshold_na):
    for variable in df.columns:
        percentage = round((df[variable].isnull().sum()/df[variable].count())*10)
        if percentage>treshold_na:
            df.drop(variable,inplace=True,axis=1)
            st.write(f"Missingness percentage of {variable} is % {percentage} and dropped.")
            drop_list.append(variable)
    if len(drop_list) == 0:
        st.write("No columns exceeding the threshold you set for missingness")
missing_drop(df,treshold_na= threshold_na)
if st.button("Generate Codes for Missing drop"):
    st.code(f"""def missing_drop(df,treshold_na):
    for variable in df.columns:
    percentage = round((df[variable].isnull().sum()/df[variable].count())*10)
    if percentage>treshold_na:
        df.drop(variable,inplace=True,axis=1)
missing_drop(df,treshold_na= {threshold_na})""")
nan_values = st.selectbox("Choose an option for Nan imputation ", ("Drop All Nan Values", "Fill Nan values median and mod"))
def impute_nan_median(df,variable):
    if df[variable].isnull().sum()>0:
        st.write(f"{variable} column has {df[variable].isnull().sum()} missing value / replaced with median:{df[variable].median()}")
        df[variable+"NAN"] = np.where(df[variable].isnull(), 1, 0)
        df[variable] = df[variable].fillna(df[variable].median())
def impute_nan_cat_mode(df,variable):
    if df[variable].isnull().sum() > 0:
        st.write(f"{variable} column has {df[variable].isnull().sum()} missing value / replaced {df[variable].mode()[0]}")

        df[variable+"NAN"] = np.where(df[variable].isnull(), 1, 0)
        frequent=df[variable].mode()[0]
        df[variable].fillna(frequent, inplace=True)
if nan_values == "Drop All Nan Values":
    for variable in df.columns:
        if df[variable].isnull().sum() > 0:
            st.write(f"{variable} column has {df[variable].isnull().sum()} missing value")
    st.write(f" percentages of total data :% {(df.isnull().sum().sum() / df.shape[0]) * 10}")
    df.dropna(inplace=True)

else:
    for i in df.columns:
        if (np.dtype(df[i])=="object"):
            impute_nan_cat_mode(df, i)
        else:
            impute_nan_median(df, i)
if st.button("Generate Codes for Missing Values"):
    if nan_values == "Drop All Nan Values":
        st.code(f""" #Drop All Nan Values
df.dropna(inplace=True)""")
    else:
        st.code(f"""def impute_nan_median(df,variable):
    if df[variable].isnull().sum()>0:
        df[variable+"NAN"] = np.where(df[variable].isnull(), 1, 0)
        df[variable] = df[variable].fillna(df[variable].median())
def impute_nan_cat_mode(df,variable):
    if df[variable].isnull().sum() > 0:
        df[variable+"NAN"] = np.where(df[variable].isnull(), 1, 0)
        frequent=df[variable].mode()[0]
        df[variable].fillna(frequent, inplace=True)
    for i in df.columns:
        if (np.dtype(df[i])=="object"):
            impute_nan_cat_mode(df, i)
        else:
            impute_nan_median(df, i)""")

st.markdown(30*"--")
st.write("""
## Outliers
""")
try:
    Outliers_handle=st.selectbox("Select an option for Outliers ",("Keep Outliers","Handle Outliers"))
    def outliers_gaussion(df,variable):
        upper_boundary = df[variable].mean()+3*df[variable].std()
        lower_boundary= df[variable].mean()-3*df[variable].std()
        df[variable] = np.where(df[variable]>upper_boundary,upper_boundary,df[variable])
        df[variable] = np.where(df[variable]<lower_boundary,lower_boundary,df[variable])
        return df[variable].describe()
    def outliers_skewed(df,variable):
        IQR = df[variable].quantile(0.75)-df[variable].quantile(0.25)
        lower_bridge = df[variable].quantile(0.25)-(IQR*1.5)
        upper_bridge = df[variable].quantile(0.75)+(IQR*1.5)
        df[variable] = np.where(df[variable]>upper_bridge,upper_bridge,df[variable])
        df[variable] = np.where(df[variable]<lower_bridge,lower_bridge,df[variable])
        return df[variable].describe()
    if Outliers_handle == "Handle Outliers":

        for i in numeric_cols:

            IQR = df[i].quantile(0.75) - df[i].quantile(0.25)
            lower_bridge = df[i].quantile(0.25) - (IQR * 1.5)
            upper_bridge = df[i].quantile(0.75) + (IQR * 1.5)
            num_outliers = df[~df[i].between(lower_bridge, upper_bridge)].value_counts().sum()
            if (df[i].max()>upper_bridge) | (df[i].min()<lower_bridge):
                        if ((df[i].skew()<2)&(df[i].skew()>0)):
                            #fig, ax = plt.subplots()
                            #sns.boxplot(df[i])
                            #st.pyplot(fig)
                            outliers_gaussion(df, i)
                            st.write(f"{i}  column has skewed and {num_outliers} outliers value.")


                        else:
                            #fig, ax = plt.subplots()
                            #sns.boxplot(df[i])
                            #st.pyplot(fig)
                            outliers_skewed(df, i)
                            st.write(f"{i}  column has gaussian distribution and {num_outliers} outliers value.")


    else:
        st.write("You are keeping all outliers")
        for i in numeric_cols:

            IQR = df[i].quantile(0.75) - df[i].quantile(0.25)
            lower_bridge = df[i].quantile(0.25) - (IQR * 1.5)
            upper_bridge = df[i].quantile(0.75) + (IQR * 1.5)
            num_outliers = df[~df[i].between(lower_bridge, upper_bridge)].value_counts().sum()
            if (df[i].max()>upper_bridge) | (df[i].min()<lower_bridge):
                st.write(f"{i} column has {num_outliers} outliers")
except:
    pass
if Outliers_handle == "Handle Outliers":
    if st.button("Generate Codes for Outliers"):
        st.code(f"""
def outliers_gaussion(df,variable):
    upper_boundary = df[variable].mean()+3*df[variable].std()
    lower_boundary= df[variable].mean()-3*df[variable].std()
    df[variable] = np.where(df[variable]>upper_boundary,upper_boundary,df[variable])
    df[variable] = np.where(df[variable]<lower_boundary,lower_boundary,df[variable])
    return df[variable].describe()
def outliers_skewed(df,variable):
    IQR = df[variable].quantile(0.75)-df[variable].quantile(0.25)
    lower_bridge = df[variable].quantile(0.25)-(IQR*1.5)
    upper_bridge = df[variable].quantile(0.75)+(IQR*1.5)
    df[variable] = np.where(df[variable]>upper_bridge,upper_bridge,df[variable])
    df[variable] = np.where(df[variable]<lower_bridge,lower_bridge,df[variable])
    return df[variable].describe()
for i in numeric_cols:
    IQR = df[i].quantile(0.75) - df[i].quantile(0.25)
    lower_bridge = df[i].quantile(0.25) - (IQR * 1.5)
    upper_bridge = df[i].quantile(0.75) + (IQR * 1.5)
    num_outliers = df[~df[i].between(lower_bridge, upper_bridge)].value_counts().sum()
    if (df[i].max()>upper_bridge) | (df[i].min()<lower_bridge):
        if ((df[i].skew()<2)&(df[i].skew()>0)):
            outliers_gaussion(df, i)
        else:
            outliers_skewed(df, i)""")
st.markdown(30*"--")
st.write("""
## Encoding
""")

y = df.loc[:,target_column]
X =df.drop([target_column],axis=1)
df=X
encode_list = []
def allonehotencoding(df):

    for i in X.columns:
        if (np.dtype(df[i]) == "object"):
            unique_value=len(df[i].value_counts().sort_values(ascending=False).head(10).index)
            if unique_value >10:
                for categories in (df[i].value_counts().sort_values(ascending=False).head(10).index):
                    df[i+"_"+categories]=np.where(df[i]==categories,1 ,0)
                    encode_list.append(i + "_" + categories)

            else:
                for categories in (df[i].value_counts().sort_values(ascending=False).head(unique_value-1).index):
                    df[i + "_" + categories] = np.where(df[i] == categories, 1, 0)
                    encode_list.append(i + "_" + categories)

    return df,encode_list
num_cat_col=len(df.select_dtypes(include=["object"]).columns)
allonehotencoding(df)
for i in df.columns:
    if (np.dtype(df[i]) == "object"):
        df = df.drop([i], axis=1)
col_after_endoded_all=df.columns
st.write(f"Onehotencoding : {num_cat_col} columns encoded and  {len(encode_list)} columns added")
if st.button("Generate Codes for Encoding"):
    st.code("""
y = df.loc[:,target_column]
df =df.drop([target_column],axis=1)
def allonehotencoding(df):
    for i in X.columns:
        if (np.dtype(df[i]) == "object"):
            unique_value=len(df[i].value_counts().sort_values(ascending=False).index)
            if unique_value >10:
                for categories in (df[i].value_counts().sort_values(ascending=False).head(10).index):
                    df[i+"_"+categories]=np.where(df[i]==categories,1 ,0)
            else:
                for categories in (df[i].value_counts().sort_values(ascending=False).head(unique_value-1).index):
                    df[i + "_" + categories] = np.where(df[i] == categories, 1, 0)
    return df
allonehotencoding(df)
for i in df.columns:
    if (np.dtype(df[i]) == "object"):
        df = df.drop([i], axis=1)""")

st.markdown(30*"--")
st.write("""
## Feature Selection
""")

Feature_importance=st.selectbox("Feature Selection",("Keep all Feature", "Reduce Features"))


def Univariate_Selection(indipendant, dependent, N):
    select_obj = SelectKBest(score_func=chi2, k=10)
    selected = select_obj.fit(indipendant, dependent)
    feature_score = pd.DataFrame(selected.scores_, index=indipendant.columns, columns=["Score"])["Score"].sort_values(
        ascending=False).reset_index()
    st.write("Feature Importance Table")
    st.table(feature_score.head(N))
    plt.figure(figsize=(16, 16))
    sns.barplot(data=feature_score.sort_values(by='Score', ascending=False).head(N), x='Score', y='index')
    plt.title(f'{N} TOP feature importance ')
    st.pyplot()
    X_Selected.extend(feature_score["index"].head(N))
def Univariate_Selection1(indipendant, dependent, N):
    select_obj = SelectKBest(score_func=chi2, k=10)
    selected = select_obj.fit(indipendant, dependent)
    feature_score = pd.DataFrame(selected.scores_, index=indipendant.columns, columns=["Score"])["Score"].sort_values(
        ascending=False).reset_index()
    st.write("Feature Importance Table")
    st.table(feature_score)

if Feature_importance == "Reduce Features":
    st.write()

    feature_number=st.slider("Select Number of Feature , SEE the feature importances table ",1,df.shape[1])
    X_Selected = []
    Univariate_Selection(df,y,feature_number)

    df = df[X_Selected]
    if st.button("Generate Code for Feature Selection"):
        if Feature_importance == "Reduce Features":
            st.code(f"""

def Univariate_Selection( independent, dependent, N):
    X_Selected=[]
    select_obj = SelectKBest(score_func=chi2, k=10)
    selected = select_obj.fit( independent, dependent)
    feature_score = pd.DataFrame(selected.scores_, index= independent.columns, columns=["Score"])["Score"].sort_values(ascending=False).reset_index()
    plt.figure(figsize=(16, 16))
    sns.barplot(data=feature_score.sort_values(by='Score', ascending=False).head(N), x='Score', y='index')
    plt.title(f'Top {feature_number} feature importance ')
    plt.show()
    X_Selected.extend(feature_score["index"].head(N))
    df = df[X_Selected] 
    return df
df = Univariate_Selection(df,y,{feature_number})""")


else :
    st.write(f"You select all features / Total Columns {len(df.columns)}")
    Univariate_Selection1(df,y,(df.columns))


st.markdown(30*"--")
st.write("""
## Standardization
""")

standard_apply = st.selectbox("Standardization",("Do Not Apply Standardization","Apply Standardization"))
if standard_apply=="Apply Standardization":
    scale_g = st.selectbox("Select technic for Gaussian distributed columns ",("StandardScaler","MinMaxScaler","RobustScaler"))
    scale_s = st.selectbox("Select technic for Skewed distributed columns ",("Logarithmic Transformation", "Box Cox Transformation", "Exponential Transformation"))
    g_cols=[]
    s_cols=[]
    num_coll=[]
    df__test=df_test.drop(target_column,axis=1)
    cate_colss=df__test.select_dtypes(exclude=["object"]).columns
    for i in cate_colss:
        if i not in (df.columns):
            pass
        else:
            num_coll.append(i)



    for i in num_coll:
        skewness=df[i].skew()
        if (skewness<2) & (skewness>0.1):
            g_cols.append(i)
        else:
            s_cols.append(i)
    for i in g_cols:
            if scale_g=="StandardScaler":
               scaler_obj=StandardScaler()
               df[g_cols]=scaler_obj.fit_transform(df[g_cols])

            elif scale_g=="MinMaxScaler":
                 scaler_obj = MinMaxScaler().fit(df[g_cols])
                 df[g_cols] = scaler_obj.transform(df[g_cols])

            else:
                scaler_obj = RobustScaler().fit(df[g_cols])
                df[g_cols] = scaler_obj.transform(df[g_cols])
    st.write(f"{g_cols} are scaled by using {scale_g}")
    for i in s_cols:
        if scale_s=="Logarithmic Transformation":
             df[i]=np.log(df[i].replace(0,0.01))

        elif scale_s == "Box Cox Transformation":

             df[i],parameters=stat.boxcox(df[i].replace(0,0.01))

        else:
             df[i]=df[i]**(1/1.2)
    st.write(f"{s_cols} are scaled by using {scale_s}")
    if st.button("Generate Codes for Standardization"):
        if standard_apply == "Apply Standardization":
            st.code(f"""
g_cols=[]
s_cols=[]
num_coll=[]
numeric_cols=df_test.select_dtypes(exclude=["object"]).columns
for i in numeric_cols:
    if i not in (df.columns):
        pass
    else:
        num_coll.append(i)
    for i in num_coll:
        skewness=df[i].skew()
        if (skewness<2) & (skewness>0.1):
            g_cols.append(i)
        else:
            s_cols.append(i)""")

            if scale_g == "StandardScaler":
                st.code(f"""#{scale_g}
scaler_obj=StandardScaler()
df[g_cols]=scaler_obj.fit_transform(df[g_cols])""")
            elif scale_g == "MinMaxScaler":
                st.code(f"""#{scale_g}
scaler_obj = MinMaxScaler().fit(df[g_cols])
df[g_cols] = scaler_obj.transform(df[g_cols])""")
            else:
                st.code(f"""#{scale_g}
scaler_obj = RobustScaler().fit(df[g_cols])
df[g_cols] = scaler_obj.transform(df[g_cols])""")
            if scale_s == "Logarithmic Transformation":
                st.code(f"""#{scale_s}
for i in s_cols:
    df[i]=np.log(df[i].replace(0,0.01))""")
            elif scale_s == "Box Cox Transformation":
                st.code(f"""#{scale_s}
for i in s_cols:
    df[i],parameters=stat.boxcox(df[i].replace(0,0.01))""")
            else:
                st.code(f"""#{scale_s}
for i in s_cols:
    df[i]=df[i]**(1/1.2)""")

st.markdown(30*"--")
st.write("""
## Principal Component Analysis
""")
use_pca = st.selectbox("Select an option Principal Component Analysis",("Do not use principal component analysis","Use principal component analysis"))
if use_pca == "Use principal component analysis":
    try:
        pca2 = PCA(n_components=df.shape[1]).fit(df)

        fig = plt.figure()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.plot(np.cumsum(pca2.explained_variance_ratio_))
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Variance Ratio")
        st.pyplot(fig)
    except:
        pass

    pca_num = st.slider("Number of Components",1,df.shape[1])
    pca1=PCA(n_components=pca_num)
    df=pca1.fit_transform(df)
    df = pd.DataFrame(df)
    st.write(f"Number of Components : {pca_num}   representation ratio : {pca1.explained_variance_ratio_.sum()}")
    if st.button("Generate Codes for PCA"):
        if use_pca == "Use principal component analysis":
            st.code(f"""pca1=PCA(n_components={pca_num})
df=pca1.fit_transform(df)
df = pd.DataFrame(df)""")



st.markdown(30*"--")
st.write("## Machine Learning")
params = dict()
models_name = st.sidebar.selectbox("Select Model", ("KNN","Logistic Regression", "SVM", "Random Forest","XGBoost","LightGBM"))

def add_parameter_ui(clf_name):

    if clf_name == "KNN":
        st.sidebar.write("Select Parameters")
        K = st.sidebar.slider("n_neighbors", 1, 15)
        params["K"] = K
        leaf_size = st.sidebar.slider("leaf_size",1,50)
        params["leaf_size"]=leaf_size
        p =st.sidebar.slider("p",1,2)
        params["p"]=p

    elif clf_name == "Logistic Regression":
        st.sidebar.write("Select Parameters")
        C = st.sidebar.slider("C", 0.0001, 1000.0)
        penalty = st.sidebar.selectbox("penalty", ('l1', 'l2'))
        max_iter = st.sidebar.slider('max_iter',1,800)
        solver = st.sidebar.selectbox("solver",('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
        params["C"] = C
        params["penalty"] = penalty
        params["max_iter"] = max_iter
        params["solver"] = solver

    elif clf_name == "SVM":
        st.sidebar.write("Select Parameters")
        C = st.sidebar.slider("C", 0.0001, 1000.0)
        st.sidebar.write(C)
        params["C"] = C
        gamma_range = st.sidebar.slider("Gamma", 0.0001, 1000.0)
        st.sidebar.write(gamma_range)
        params["gamma"]=gamma_range


    elif clf_name == "XGBoost":
        st.sidebar.write("Select Parameters")
        max_depth = st.sidebar.slider("max_depth", 10, 200)
        n_estimators = st.sidebar.slider("n_estimators", 50, 2000)
        subsample = st.sidebar.slider("subsample", 0.01,0.90)
        learning_rate = st.sidebar.slider("learning_rate", 0.001,0.999)
        colsample_bytree = st.sidebar.slider("colsample_bytree", 0.01,0.90)
        min_child_weight = st.sidebar.slider("min_child_weight", 1, 5)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["min_child_weight"] = min_child_weight
        params["colsample_bytree"] = colsample_bytree
        params["learning_rate"] = learning_rate
        params["subsample"] = subsample
    elif clf_name == "LightGBM":
        st.sidebar.write("Select Parameters")
        max_depth = st.sidebar.slider("max_depth", 5, 200)
        reg_alpha = st.sidebar.slider("reg_alpha", 1, 20)
        n_estimators = st.sidebar.slider("n_estimators", 50, 2000)
        boosting_type = st.sidebar.selectbox("boosting_type", ('gbdt', 'dart', 'goss'))
        reg_lambda = st.sidebar.slider("reg_lambda", 1,20)
        num_leaves = st.sidebar.slider("num_leaves", 2, 300)
        colsample_bytree = st.sidebar.slider("colsample_bytree", 0.01,0.90)
        min_child_samples = st.sidebar.slider("min_child_samples", 1, 5)
        min_child_weight = st.sidebar.slider("min_child_weight", 1, 5)
        learning_rate = st.sidebar.slider("learning_rate", 0.001, 0.999)
        params["max_depth"] =  max_depth
        params["reg_alpha"] = reg_alpha
        params["n_estimators"] = n_estimators
        params["boosting_type"] = boosting_type
        params["reg_lambda"] = reg_lambda
        params["num_leaves"] = num_leaves
        params["colsample_bytree"] = colsample_bytree
        params["min_child_weight"] = min_child_weight
        params["min_child_samples"] = min_child_samples
        params["learning_rate"] = learning_rate


    else:

        st.sidebar.write("Select Parameters")
        max_depth = st.sidebar.slider("max_depth", 10, 1000)
        n_estimators = st.sidebar.slider("n_estimators", 50, 2000)
        criterion = st.sidebar.selectbox("criterion",('entropy','gini'))
        max_features = st.sidebar.selectbox("max_features",('auto', 'sqrt','log2') )
        min_samples_leaf = st.sidebar.slider("min_samples_leaf",1,10 )
        min_samples_split = st.sidebar.slider("min_samples_split",2,20 )
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        params["max_features"] = max_features
        params["min_samples_leaf"] = min_samples_leaf
        params["min_samples_split"] = min_samples_split


    return params
add_parameter_ui(models_name)

def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"],leaf_size=params["leaf_size"],p=params["p"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"],gamma=params["gamma"])
    elif clf_name =="Logistic Regression":
        clf = LogisticRegression(C=params["C"],penalty=params["penalty"],max_iter=params["max_iter"],solver=params["solver"] )
    elif clf_name == "XGBoost":
        clf = XGBClassifier(max_depth = params["max_depth"],
                            n_estimators = params["n_estimators"],
                            min_child_weight = params["min_child_weight"],
                            colsample_bytree=params["colsample_bytree"],
                            learning_rate=params["learning_rate"],
                            subsample=params["subsample"])
    elif clf_name == "LightGBM":
        clf = lgb.LGBMClassifier(**params)
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"],
                                     criterion=params["criterion"],
                                     max_features=params["max_features"],
                                     min_samples_leaf=params["min_samples_leaf"],
                                     min_samples_split=params["min_samples_split"] ,
                                     random_state=42)

    return clf
    # Classification
clf = get_classifier(models_name,params)
st.cache()
df_train = df
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.sidebar.write(f"classifier = {models_name}")
st.sidebar.write(f"accuracy = {acc}")
st.write(f"classifier = {models_name}")
st.write(f"accuracy = {acc}")
if st.button("Generate Codes for Machine Learning"):
    if models_name == "KNN":
        st.code(f"""clf = KNeighborsClassifier(n_neighbors={params["K"]},leaf_size={params["leaf_size"]},p={params["p"]})""")
    elif models_name == "SVM":
        st.code(f"""clf = SVC(C={params["C"]},gamma={params["gamma"]}""")
    elif models_name == "Logistic Regression":
        st.code(f"""
clf = LogisticRegression(C={params["C"]},
                         penalty="{params["penalty"]}",
                         max_iter={params["max_iter"]},
                         solver="{params["solver"]}")""")
    elif models_name == "XGBoost":
        st.code(f"""
clf = XGBClassifier(max_depth={params["max_depth"]},
                    n_estimators={params["n_estimators"]},
                    min_child_weight={params["min_child_weight"]},
                    colsample_bytree={params["colsample_bytree"]},
                    learning_rate={params["learning_rate"]},
                    subsample={params["subsample"]})""")
    elif models_name == "LightGBM":
        st.code(f"""
clf = lgb.LGBMClassifier(n_estimators={params["n_estimators"]},
                    boosting_type={params["boosting_type"]},
                    num_leaves={params["num_leaves"]},
                    reg_alpha={params["reg_alpha"]},
                    reg_lambda={params["reg_lambda"]},
                    colsample_bytree={params["colsample_bytree"]},
                    min_child_weight={params["min_child_weight"]},
                    min_child_samples={params["min_child_samples"]})""")
    else:
        st.code(f"""
clf = RandomForestClassifier(n_estimators={params["n_estimators"]}, 
                                max_depth={params["max_depth"]},
                                criterion={params["criterion"]},
                                max_features={params["max_features"]},
                                min_samples_leaf={params["min_samples_leaf"]},
                                min_samples_split={params["min_samples_split"]} ,
                                random_state=42)""")
    st.code(f"""X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2,random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)""")

st.write(""" ## Download Model """)
st.cache()
def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model .pkl File</a> (right-click and save as &lt;model&gt;.pkl)'
    st.markdown(href, unsafe_allow_html=True)
download_model(clf)
# save the model to disk
if st.button("Generate Code for Save & Load Model"):
    st.code("""#save & load the model  
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))""")
# PLOT


try:
    st.write(""" ## Visualization of the  Model """)
    pca = PCA(2)
    X_projected = pca.fit_transform(df)
    x1 = X_projected[:,0]
    x2 = X_projected[:,1]

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    st.pyplot()


except:
    pass
if st.button("Generate Code for Visualization"):
    st.code("""pca = PCA(2)
X_projected = pca.fit_transform(df)
x1 = X_projected[:,0]
x2 = X_projected[:,1]
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()""")
st.cache()

df_test.drop(target_column, axis= 1, inplace=True)
columns = df_test.columns
print(columns)
cat_columns = df_test.select_dtypes(include=["object"]).columns
st.write(""" ## PREDICTION """)
try:
    test_file = st.file_uploader("Upload Prediction CSV file ", type="csv", accept_multiple_files=False)
    dftest = pd.read_csv(test_file)
    testfile = dftest
except:

    st.write("OR ENTER YOUR DATA MANUALLY")
    dftest = pd.DataFrame()

    testfile = dftest
    for i in df_test.columns:
        if i not in cat_columns:
            try:
                dftest[i] = [int(st.text_input(i))]
            except:
                dftest[i] = np.nan



        else :
            if i not in ["name", "Name", "Surname"]:
                dftest[i] = [st.selectbox(i, df_test[i].value_counts().index.tolist())]

            else :
                try:
                    dftest[i] = [st.text_input(i)]
                except :
                    dftest[i] = np.nan


st.write(dftest)


def Prediction(dftest,testfile):
    ## Correlation Drop
    try:
        dftest.drop(corCol, axis=1, inplace=True)
    except:
        pass
    ## Missing Drop
    try:
        dftest.drop(drop_list, axis=1, inplace=True)
    except:
        pass

    def impute_nan_median(df, variable):
        if df[variable].isnull().sum() > 0:
            df[variable + "NAN"] = np.where(df[variable].isnull(), 1, 0)
            df[variable] = df[variable].fillna(df_test[variable].median())

    def impute_nan_cat_mode(df, variable):
        if df[variable].isnull().sum() > 0:
            df[variable + "NAN"] = np.where(df[variable].isnull(), 1, 0)
            frequent = df_test[variable].mode()[0]
            df[variable].fillna(frequent, inplace=True)

    for i in dftest.columns:
        if (np.dtype(dftest[i]) == "object"):
            impute_nan_cat_mode(dftest, i)
        else:
            impute_nan_median(dftest, i)

    ## Outliers

    def outliers_gaussion(df, variable):
        upper_boundary = df_test[variable].mean() + 3 * df[variable].std()
        lower_boundary = df_test[variable].mean() - 3 * df[variable].std()
        df[variable] = np.where(df[variable] > upper_boundary, upper_boundary, df[variable])
        df[variable] = np.where(df[variable] < lower_boundary, lower_boundary, df[variable])
        return df[variable].describe()

    def outliers_skewed(df, variable):
        IQR = df_test[variable].quantile(0.75) - df_test[variable].quantile(0.25)
        lower_bridge = df_test[variable].quantile(0.25) - (IQR * 1.5)
        upper_bridge = df_test[variable].quantile(0.75) + (IQR * 1.5)
        df[variable] = np.where(df[variable] > upper_bridge, upper_bridge, df[variable])
        df[variable] = np.where(df[variable] < lower_bridge, lower_bridge, df[variable])
        return df[variable].describe()
    try:
        if Outliers_handle == "Handle Outliers":
            for i in numeric_cols:
                outliers_gaussion(dftest, i)
    except:
        pass

    ## One hot encoding

    def allonehotencoding_test(df):
        for i in encode_list:
            try:
                for categories in df[i]:
                    df[i] = np.where(df[i] == categories, 1, 0)
            except:
                df[i] = np.where(False, 1, 0)

        return df

    allonehotencoding_test(dftest)
    for i in col_after_endoded_all:
        if i not in dftest.columns:
            dftest[i]=np.where(False,1,0)

    dftest = dftest.loc[:,col_after_endoded_all]
    dftest = dftest.drop(dftest.select_dtypes("object").columns, axis=1)

    ## feature importance

    if Feature_importance == "Reduce Features":
        dftest = dftest[X_Selected]
    ## Standardization
    if standard_apply == "Apply Standardization":
        try:
          dftest[g_cols] = scaler_obj.fit_transform(dftest[g_cols])
        except:
            pass

        for i in s_cols:
            if scale_s == "Logarithmic Transformation":
                dftest[i] = np.log(dftest[i].replace(0, 0.01))

            elif scale_s == "Box Cox Transformation":
                try:
                    dftest[i], parameters = stat.boxcox(dftest[i].replace(0, 0.01))
                except:
                    pass
            else:
                dftest[i] = dftest[i] ** (1 / 1.2)
    ##PCA
    if use_pca == "Use principal component analysis":
        dftest = pca1.transform(dftest)
        dftest = pd.DataFrame(dftest)
    ## Prediction
    try:
        result = clf.predict(dftest)
        st.write(f"result {result.iloc[:,0]}")


    except:
        result = clf.predict(dftest)
        output = pd.DataFrame()
        outputkaggle=pd.DataFrame()
        output[target_column]=np.array(result)
        for i in testfile:
            output[i]=testfile[i]
        st.write(output)
        col_id = list(testfile.columns)[0]
        outputkaggle[col_id]=testfile.iloc[:,0]
        outputkaggle[target_column]=np.array(result)




        def get_table_download_link(df):

            csv = df.to_csv(index=False)
            b64 = base64.b64encode(
                csv.encode()
            ).decode()  # some strings <-> bytes conversions necessary here
            return f'<a href="data:file/csv;base64,{b64}" download="Output.csv">Download Output csv file</a>'

        st.markdown(get_table_download_link(output), unsafe_allow_html=True)
        st.cache()
        try:

            def get_table_download_link_kaggle(df):

                csv = df.to_csv(index=False)
                b64 = base64.b64encode(
                    csv.encode()
                ).decode()  # some strings <-> bytes conversions necessary here
                return f'<a href="data:file/csv;base64,{b64}" download="Submission.csv">Download Submission csv file for Kaggle </a>'

            st.markdown(get_table_download_link_kaggle(outputkaggle), unsafe_allow_html=True)
        except:
            pass



if st.button("Predict"):
    Prediction(dftest,testfile)

st.write("https://github.com/MustafaBozkurt84/MustafaBozkurt84-heroku_machine_learning_app")









