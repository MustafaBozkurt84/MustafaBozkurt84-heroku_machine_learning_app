import streamlit as st
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
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
from sklearn.model_selection import cross_val_score,KFold, GroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from tpot import TPOTClassifier
import optuna
import joblib
import os
import streamlit.components.v1 as components
import marshal, types
st.sidebar.write("https://github.com/MustafaBozkurt84")
st.sidebar.write("https://www.linkedin.com/in/mustafa-bozkurt-3405a91a5/")
html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Create Your Own Model and Generate Code</h2>
    </div>
    """

st.markdown(html_temp, unsafe_allow_html = True)

html_temp1 = """
    <div style="background:#025246 ;padding:1px">
    <h3 style="color:white;text-align:left;"> With this app, you can create and test your own model with your own data</h3>
    </div>
    """
st.markdown(html_temp1, unsafe_allow_html = True)
html_temp2 = """
    <div style="padding:10px">
    <h5 style="color:red;text-align:left;"> Currently only works for Classification Models</h5>
    </div>
    """
st.markdown(html_temp2, unsafe_allow_html = True)
st.image("stickman.gif")

PAGE =  st.sidebar.radio("Select Page ",["Feature Engineering and Machine Learning","Prediction and Testing","Deployment"])
if  PAGE == "Feature Engineering and Machine Learning":

    st.markdown("""
    <div style="padding:10px">
    <h4 style="color:black;text-align:left;"> Let's start !</h4>
    </div>
    """, unsafe_allow_html = True)
    load_data_html = """
        <div style="background:#025246 ;padding:10px">
        <h4 style="color:white;text-align:left;"> Load your own dataset or select dataset</h4>
        </div>
        """
    st.markdown(load_data_html, unsafe_allow_html = True)



    uploaded_files = st.file_uploader("Upload new dataset CSV ", type="csv", accept_multiple_files=False)

    try:
        df = pd.read_csv(uploaded_files)
        df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        target_column = st.selectbox("Select your target Column", df.columns)
    except:
        dataset_name = st.selectbox("Select Datasets", ("Titanic","Churn"))
        st.markdown(f"""
<!DOCTYPE html>
<title>My Example</title>
<style>
	.box {{
		background-color: transparent;
		font-size: 1vw;
		padding: 0vw;
		margin: 1vw;
		border: solid;
        border-color:#F50057;
	}}
</style>

<div class="box">{dataset_name}</div>""", unsafe_allow_html=True)




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
        eda_html = """
              <div style="background:#025246 ;padding:10px">
              <h4 style="color:white;text-align:left;"> exploratory data analysis</h4>
              </div>
              """
        st.markdown(eda_html, unsafe_allow_html=True)
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
    st.markdown("""
        <div style="background:#025246 ;padding:10px">
        <h3 style="color:white;text-align:left;"> Correlation (Data Cleaning)</h3>
        </div>
        """,unsafe_allow_html = True)

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

        st.markdown(f"""<div style="color:black";border-color:#F50057" class="box">{corCol} columns correlations more than {threshold} and we dropped</div>""", unsafe_allow_html=True)

    else:
        st.markdown("""<div style="color:black;border-color:black;" class="box">No columns exceeding the threshold you set for correlation</div>""", unsafe_allow_html=True)



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
    st.markdown("""
            <div style="background:#025246 ;padding:10px">
            <h3 style="color:white;text-align:left;"> Missing Imputation</h3>
            </div>
            """, unsafe_allow_html=True)

    threshold_na = st.selectbox("Select threshold for the percentage of missingness and remove these features", (100, 90, 80, 70, 60, 50,40,30,10))
    drop_list = []
    def missing_drop(df,treshold_na):
        for variable in df.columns:
            percentage = round((df[variable].isnull().sum()/df[variable].count())*10)
            if percentage>treshold_na:
                df.drop(variable,inplace=True,axis=1)
                st.markdown(
                    f"""<div style="color:black";border-color:#F50057" class="box">Missingness percentage of {variable} is % {percentage} and dropped.</div>""",unsafe_allow_html=True)

                drop_list.append(variable)
        if len(drop_list) == 0:
            st.markdown(
                """<div style="color:black;border-color:black;" class="box">No columns exceeding the threshold you set for missingness</div>""",unsafe_allow_html=True)



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
            st.markdown(
                f"""<div style="color:black";border-color:#F50057" class="box">{variable} column has {df[variable].isnull().sum()} missing value / replaced with median:{df[variable].median()}</div>""",unsafe_allow_html=True)

            df[variable+"NAN"] = np.where(df[variable].isnull(), 1, 0)
            df[variable] = df[variable].fillna(df[variable].median())
    def impute_nan_cat_mode(df,variable):
        if df[variable].isnull().sum() > 0:
            st.markdown(
                f"""<div style="color:black";border-color:#F50057" class="box">{variable} column has {df[variable].isnull().sum()} missing value / replaced {df[variable].mode()[0]}</div>""",unsafe_allow_html=True)

            df[variable+"NAN"] = np.where(df[variable].isnull(), 1, 0)
            frequent=df[variable].mode()[0]
            df[variable].fillna(frequent, inplace=True)
    if nan_values == "Drop All Nan Values":
        for variable in df.columns:
            if df[variable].isnull().sum() > 0:
                st.markdown(
                    f"""<div style="color:black";border-color:#F50057" class="box">{variable} column has {df[variable].isnull().sum()} missing value</div>""",unsafe_allow_html=True)
        st.markdown(
            f"""<div style="color:black";border-color:#F50057" class="box">percentages of total data :% {(df.isnull().sum().sum() / df.shape[0]) * 10}</div>""",unsafe_allow_html=True)
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
    st.markdown("""
                <div style="background:#025246 ;padding:10px">
                <h3 style="color:white;text-align:left;"> Outliers</h3>
                </div>
                """, unsafe_allow_html=True)

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
                                st.markdown(
                                    f"""<div style="color:black";border-color:#F50057" class="box">{i}  column has skewed and {num_outliers} outliers value.</div>""",unsafe_allow_html=True)
                            else:
                                #fig, ax = plt.subplots()
                                #sns.boxplot(df[i])
                                #st.pyplot(fig)
                                outliers_skewed(df, i)
                                st.markdown(f"""<div style="color:black";border-color:#F50057" class="box">{i}  column has gaussian distribution and {num_outliers} outliers value.</div>""",unsafe_allow_html=True)

        else:
            st.markdown(
                """<div style="color:black;border-color:black;" class="box">You are keeping all outliers</div>""",unsafe_allow_html=True)

            for i in numeric_cols:

                IQR = df[i].quantile(0.75) - df[i].quantile(0.25)
                lower_bridge = df[i].quantile(0.25) - (IQR * 1.5)
                upper_bridge = df[i].quantile(0.75) + (IQR * 1.5)
                num_outliers = df[~df[i].between(lower_bridge, upper_bridge)].value_counts().sum()
                if (df[i].max()>upper_bridge) | (df[i].min()<lower_bridge):
                    st.markdown(
                        f"""<div style="color:black;border-color:black;" class="box">{i} column has {num_outliers} outliers</div>""",unsafe_allow_html=True)

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
    st.markdown("""
                <div style="background:#025246 ;padding:10px">
                <h3 style="color:white;text-align:left;">Encoding</h3>
                </div>
                """, unsafe_allow_html=True)


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
                        df[i+categories]=np.where(df[i]==categories,1 ,0)
                        encode_list.append(i +  categories)

                else:
                    for categories in (df[i].value_counts().sort_values(ascending=False).head(unique_value-1).index):
                        df[i + categories] = np.where(df[i] == categories, 1, 0)
                        encode_list.append(i  + categories)

        return df,encode_list
    num_cat_col=len(df.select_dtypes(include=["object"]).columns)
    allonehotencoding(df)
    for i in df.columns:
        if (np.dtype(df[i]) == "object"):
            df = df.drop([i], axis=1)
    col_after_endoded_all=df.columns
    st.markdown(
        f"""<div style="color:black";border-color:#F50057" class="box">Onehotencoding : {num_cat_col} columns encoded and  {len(encode_list)} columns added</div>""",
        unsafe_allow_html=True)
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
                        df[i+categories]=np.where(df[i]==categories,1 ,0)
                else:
                    for categories in (df[i].value_counts().sort_values(ascending=False).head(unique_value-1).index):
                        df[i + categories] = np.where(df[i] == categories, 1, 0)
        return df
    allonehotencoding(df)
    for i in df.columns:
        if (np.dtype(df[i]) == "object"):
            df = df.drop([i], axis=1)""")

    st.markdown(30*"--")
    st.markdown("""
                            <div style="background:#025246 ;padding:10px">
                            <h3 style="color:white;text-align:left;"> Feature Selection</h3>
                            </div>
                            """, unsafe_allow_html=True)

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
        st.markdown(f"""<div style="color:black;border-color:black;" class="box">You select all features / Total Columns {len(df.columns)}</div>""",unsafe_allow_html=True)
        Univariate_Selection1(df,y,(df.columns))


    st.markdown(30*"--")
    st.markdown("""
                        <div style="background:#025246 ;padding:10px">
                        <h3 style="color:white;text-align:left;"> Standardization</h3>
                        </div>
                        """, unsafe_allow_html=True)


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
    st.markdown("""
                    <div style="background:#025246 ;padding:10px">
                    <h3 style="color:white;text-align:left;"> Principal Component Analysis</h3>
                    </div>
                    """, unsafe_allow_html=True)

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
    models_name = st.sidebar.selectbox("Select Model",("KNN", "Logistic Regression", "SVM", "Random Forest", "XGBoost", "LightGBM"))
    st.markdown(f"""
                                <div style="background:#025246 ;padding:10px">
                                <h3 style="color:white;text-align:left;"> Machine Learning {models_name}</h3>
                                </div>
                                """, unsafe_allow_html=True)

    try:
        df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    except:
        pass
    params = dict()


    def add_parameter_ui(clf_name):

        if clf_name == "KNN":
            st.sidebar.write("Select Parameters")
            K = st.sidebar.slider("n_neighbors", 1, 15,5)
            params["K"] = K
            leaf_size = st.sidebar.slider("leaf_size",1,50,30)
            params["leaf_size"]=leaf_size
            p =st.sidebar.slider("p",1,2,2)
            params["p"]=p

        elif clf_name == "Logistic Regression":
            st.sidebar.write("Select Parameters")
            C = st.sidebar.slider("C", 0.0001, 1000.0,1.0)
            penalty = st.sidebar.selectbox("penalty", ('l2', 'l1'))
            max_iter = st.sidebar.slider('max_iter',1,800,100)
            solver = st.sidebar.selectbox("solver",('lbfgs','newton-cg','liblinear', 'sag', 'saga'))
            params["C"] = C
            params["penalty"] = penalty
            params["max_iter"] = max_iter
            params["solver"] = solver

        elif clf_name == "SVM":
            st.sidebar.write("Select Parameters")
            C = st.sidebar.slider("C", 0.0001, 1000.0,1.0)
            st.sidebar.write(C)
            params["C"] = C
            gamma_range = st.sidebar.slider("Gamma", 0.0001, 1000.0)
            st.sidebar.write(gamma_range)
            params["gamma"]=gamma_range


        elif clf_name == "XGBoost":
            st.sidebar.write("Select Parameters")
            max_depth = st.sidebar.slider("max_depth", 2, 200,3)
            n_estimators = st.sidebar.slider("n_estimators", 50, 2000,100)
            subsample = st.sidebar.slider("subsample", 0.01,1.00,1.00)
            learning_rate = st.sidebar.slider("learning_rate", 0.001,0.999,0.100)
            colsample_bytree = st.sidebar.slider("colsample_bytree", 0.01,1.0,1.0)
            min_child_weight = st.sidebar.slider("min_child_weight", 1, 5,1)
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
            params["min_child_weight"] = min_child_weight
            params["colsample_bytree"] = colsample_bytree
            params["learning_rate"] = learning_rate
            params["subsample"] = subsample
        elif clf_name == "LightGBM":
            st.sidebar.write("Select Parameters")
            max_depth = st.sidebar.slider("max_depth", -1, 200,-1)
            reg_alpha = st.sidebar.slider("reg_alpha", 0.0, 20.0,0.0)
            n_estimators = st.sidebar.slider("n_estimators", 50, 2000,100)
            boosting_type = st.sidebar.selectbox("boosting_type", ('gbdt', 'dart', 'goss'))
            reg_lambda = st.sidebar.slider("reg_lambda", 0.0,20.0,0.0)
            num_leaves = st.sidebar.slider("num_leaves", 2, 300,31)
            colsample_bytree = st.sidebar.slider("colsample_bytree", 0.01,1.0,1.0)
            min_child_samples = st.sidebar.slider("min_child_samples", 1, 30,20)
            min_child_weight = st.sidebar.slider("min_child_weight", 0.001, 1.0,0.001)
            learning_rate = st.sidebar.slider("learning_rate", 0.001, 0.999,0.1)
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
            max_depth = st.sidebar.slider("max_depth", 1, 1000,1)
            n_estimators = st.sidebar.slider("n_estimators", 50, 2000,100)
            criterion = st.sidebar.selectbox("criterion",('gini','entropy'))
            max_features = st.sidebar.selectbox("max_features",('auto', 'sqrt','log2') )
            min_samples_leaf = st.sidebar.slider("min_samples_leaf",1,10,1 )
            min_samples_split = st.sidebar.slider("min_samples_split",2,20,2 )
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

    use_cv =st.selectbox("Cross Validation",("Do not use Cross Validation","Use Cross Validation"))
    if use_cv == "Use Cross Validation":
        fold_cv = st.selectbox("Select Fold Type",("KFold","StratifiedKFold","TimeSeriesSplit"))
        n_splits_fold =st.slider("n_splits",2,10)
        if fold_cv == "KFold":
            fold = KFold(n_splits=n_splits_fold)
        elif fold_cv == "StratifiedKFold":
            fold = StratifiedKFold(n_splits=n_splits_fold)
        elif fold_cv == "TimeSeriesSplit":
            fold = TimeSeriesSplit(n_splits=n_splits_fold)
        acc = cross_val_score(clf, df, y, cv= fold, scoring="accuracy")
        st.sidebar.markdown(f"""
                    <div style="background:#025246 ;padding:1px">
                    <h3 style="color:white;text-align:left;">{models_name}</h3>
                    </div>
                    """, unsafe_allow_html=True)
        st.sidebar.markdown(f"""
            <div style="background:#025246 ;padding:1px">
            <h3 style="color:white;text-align:left;">acc mean= {acc.mean()}</h3>
            </div>
            """,unsafe_allow_html = True)
        st.markdown(f"""<div style="color:black;border-color:black;" class="box">accuracy for each fold = {acc}</div>""",unsafe_allow_html=True)
        st.markdown(f"""<div style="color:black;border-color:black;" class="box">accuracy mean = {acc.mean()}</div>""",unsafe_allow_html=True)

    else:
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.sidebar.markdown(f"""
                    <div style="background:#025246 ;padding:1px">
                    <h3 style="color:white;text-align:left;">{models_name}</h3>
                    </div>
                    """, unsafe_allow_html=True)
        st.sidebar.markdown(f"""
                            <div style="background:#025246 ;padding:1px">
                            <h3 style="color:white;text-align:left;">acc = {acc}</h3>
                            </div>
                            """, unsafe_allow_html=True)

        st.markdown(f"""
                                    <div style="background:	#757575 ;padding:1px">
                                    <h3 style="color:white;text-align:left;">acc = {acc}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
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
                        boosting_type="{params["boosting_type"]}",
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
        if use_cv == "Do not use Cross Validation":
            st.code(f"""X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2,random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)""")
        else:
            st.code(f"""acc = cross_val_score(clf, df, y, cv={fold}, scoring="accuracy")""")

    st.markdown(f"""
                                    <div style="background:#025246 ;padding:10px">
                                    <h3 style="color:white;text-align:left;"> Model Tuning for {models_name}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)

    model_tuning = st.selectbox("Select Option for Model Tuning",("Do not Apply Hyper Parameter Optimization ","Apply Hyper Parameter Optimization"))
    optimizers =["Select","RandomizedSearchCV-GridSearchCV", "Bayesian Optimization (Hyperopt)", "Optuna- Automate Hyperparameter Tuning", "Genetic Algorithms (TPOT Classifier)"]
    if model_tuning == "Apply Hyper Parameter Optimization":
        fold_cv_hyp = st.selectbox("Select Fold Type", ("KFold ", "StratifiedKFold ", "TimeSeriesSplit "))
        n_splits_fold_hyp = st.slider("n_splits ", 2, 5)
        if fold_cv_hyp == "KFold ":
            fold = KFold(n_splits=n_splits_fold_hyp)
        elif fold_cv_hyp == "StratifiedKFold ":
            fold = StratifiedKFold(n_splits=n_splits_fold_hyp)
        elif fold_cv_hyp == "TimeSeriesSplit ":
            fold = TimeSeriesSplit(n_splits=n_splits_fold_hyp)
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
        hyperop = st.selectbox("Select Optimizer",optimizers)
        params = dict()

        if models_name == "KNN":
            K = [int(x) for x in np.linspace(start=3, stop=20, num=10)]
            params["n_neighbors"] = K
            leaf_size = [int(x) for x in np.linspace(start=1, stop=50, num=10)]
            params["leaf_size"] = leaf_size
            p = [1, 2]
            params['metric'] = ['euclidean', 'manhattan']
            params["p"] = p
            iteration = 100
            clf_tune = KNeighborsClassifier()
        elif models_name == "Logistic Regression":

            C = [int(x) for x in np.linspace(start=0.001, stop=1000.0, num=10)]
            penalty = ['l2', 'l1']
            max_iter = [int(x) for x in np.linspace(start=1, stop=500, num=10)]
            solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            params["C"] = C
            params["penalty"] = penalty
            params["max_iter"] = max_iter
            params["solver"] = solver
            iteration = 200
            clf_tune = LogisticRegression()
        elif models_name == "SVM":

            C = [int(x) for x in np.linspace(start=0.001, stop=1000.0, num=10)]
            params["C"] = C
            gamma_range = [int(x) for x in np.linspace(start=0.01, stop=1000.0, num=10)]
            params["gamma"] = gamma_range
            iteration = 100
            clf_tune = SVC()
        elif models_name == "XGBoost":

            max_depth = [int(x) for x in np.linspace(start=2, stop=200, num=10)]
            n_estimators = [int(x) for x in np.linspace(start=50, stop=2000, num=10)]
            subsample = [x for x in np.linspace(start=0.01, stop=1.0, num=4)]
            learning_rate = [x for x in np.linspace(start=0.01, stop=0.90, num=4)]
            colsample_bytree = [x for x in np.linspace(start=0.01, stop=0.90, num=4)]
            min_child_weight = [int(x) for x in np.linspace(start=1, stop=5, num=4)]
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
            params["min_child_weight"] = min_child_weight
            params["colsample_bytree"] = colsample_bytree
            params["learning_rate"] = learning_rate
            params["subsample"] = subsample
            iteration = 200
            clf_tune = XGBClassifier()
        elif models_name == "LightGBM":

            max_depth = [int(x) for x in np.linspace(start=-1, stop=200, num=5)]
            reg_alpha = [x for x in np.linspace(start=0.0, stop=20.0, num=4)]
            n_estimators = [int(x) for x in np.linspace(start=50, stop=2000, num=10)]
            boosting_type = ['gbdt', 'dart', 'goss']
            reg_lambda = [x for x in np.linspace(start=0.001, stop=20.0, num=5)]
            num_leaves = [int(x) for x in np.linspace(start=2, stop=300, num=10)]
            colsample_bytree = [x for x in np.linspace(start=0.1, stop=0.9, num=5)]
            min_child_samples = [int(x) for x in np.linspace(start=1, stop=5, num=3)]
            min_child_weight = [x for x in np.linspace(start=0.001, stop=1.0, num=2)]
            learning_rate = [x for x in np.linspace(start=0.001, stop=0.999, num=3)]
            params["max_depth"] = max_depth
            params["reg_alpha"] = reg_alpha
            params["n_estimators"] = n_estimators
            params["boosting_type"] = boosting_type
            params["reg_lambda"] = reg_lambda
            params["num_leaves"] = num_leaves
            params["colsample_bytree"] = colsample_bytree
            params["min_child_weight"] = min_child_weight
            params["min_child_samples"] = min_child_samples
            params["learning_rate"] = learning_rate
            iteration = 200
            clf_tune = lgb.LGBMClassifier()

        else:


            params["n_estimators"] = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
            params["criterion"] = ['entropy', 'gini']
            params["max_features"] = ['auto', 'sqrt', 'log2']
            params["min_samples_leaf"] = [int(x) for x in np.linspace(start=1, stop=10, num=3)]
            params["min_samples_split"] = [int(x) for x in np.linspace(start=2, stop=15, num=3)]
            params["max_depth"] = [int(x) for x in np.linspace(start=1, stop=1000, num=10)]
            clf_tune = RandomForestClassifier()
            iteration = 400
        if hyperop == "RandomizedSearchCV-GridSearchCV":
            iteration = st.slider("Select iteration",100,2000,300)

            st.code(f"""#Parameter for RandomizedSearchCV 
    params = {params}""")

            #RandomizedSearchCV

            clf_randomcv = RandomizedSearchCV(estimator=clf_tune,
                                             param_distributions=params,
                                             n_iter=iteration,
                                             cv=fold,
                                             verbose=2,
                                             random_state=100,
                                             n_jobs=-1)


            clf_randomcv.fit(X_train, y_train)
            st.code(f"""#RandomizedSearchCV best parameters 
    {clf_randomcv.best_params_}""")
            st.code(f"""#RandomizedSearchCV 
    {clf_randomcv}""")
            st.code(f"""#RandomizedSearchCV best estimator 
    {clf_randomcv.best_estimator_}""")
            clf = clf_randomcv.best_estimator_
            y_pred = clf.predict(X_test)
            confusion_matrix(y_test, y_pred)
            accuracy_score(y_test, y_pred)
            classification_report(y_test, y_pred)
            st.code("confusion_matrix RandomizedSearchCV: {}".format(confusion_matrix(y_test, y_pred)))
            st.markdown(f"""
                                                <div style="background:	#757575 ;padding:1px">
                                                <h3 style="color:white;text-align:left;">Accuracy Score RandomizedSearchCV: {accuracy_score(y_test, y_pred)}</h3>
                                                </div>
                                                """, unsafe_allow_html=True)
            st.code("Classification report RandomizedSearchCV: {}".format(classification_report(y_test, y_pred)))
            st.sidebar.markdown(f"""
                        <div style="background:#025246 ;padding:1px">
                        <h3 style="color:white;text-align:left;">Accuracy Score RandomizedSearchCV : {accuracy_score(y_test, y_pred)}</h3>
                        </div>
                        """, unsafe_allow_html=True)


            st.cache()

            #if st.sidebar.button("Continue GridSearchCV"):
            st.write("### GridSearchCV")
            grid_params = dict()
            for key, value in clf_randomcv.best_params_.items():
                if type(value) == str:
                    grid_params[key] = [value]
                elif type(value) == float:
                    grid_params[key] = [value, 2 * value, 3 * value, value / 2, value / 3]
                elif value > 100:
                    grid_params[key] = [value, value + 20, value - 20, value + 40, value - 40]
                elif (value > 10) & (value < 100):
                    grid_params[key] = [value, value + 5, value - 5, value + 3, value - 3]
                elif value < 10:
                    grid_params[key] = [value, value + 1, value - 1]
                elif value == 1:
                    grid_params[key] = [value, value + 1]
                elif value == 2:
                    grid_params[key] = [value, value + 1, value + 2]


            grid_search = GridSearchCV(estimator=clf_tune, param_grid=grid_params, cv=fold, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)

            clf_grid = grid_search.best_estimator_
            y_pred = clf_grid.predict(X_test)
            clf = clf_grid
            confusion_matrix(y_test, y_pred)
            accuracy_score(y_test, y_pred)
            classification_report(y_test, y_pred)

            st.code(f"""#GridSearchCV 
    {grid_search}""")
            st.code(f"""#GridSearchCV best estimator 
    {grid_search.best_estimator_}""")
            st.code("confusion_matrix GridSearchCV: {}".format(confusion_matrix(y_test, y_pred)))
            st.code(
                "Classification report GridSearchCV: {}".format(classification_report(y_test, y_pred)))
            st.markdown(f"""
                                                <div style="background:	#757575 ;padding:1px">
                                                <h3 style="color:white;text-align:left;">Accuracy Score GridSearchCV {accuracy_score(y_test, y_pred)}</h3>
                                                </div>
                                                """, unsafe_allow_html=True)

            st.sidebar.markdown(f"""
                        <div style="background:#025246 ;padding:1px">
                        <h3 style="color:white;text-align:left;">Acc GridSearchCV {accuracy_score(y_test, y_pred)}</h3>
                        </div>
                        """, unsafe_allow_html=True)



            st.cache()
        if hyperop == "Bayesian Optimization (Hyperopt)":
                params = dict()
                if models_name == "KNN":
                    params = dict()
                    params["n_neighbors"] = hp.uniform("n_neighbors", 2, 50)
                    params["leaf_size"] = hp.uniform("leaf_size", 1, 50)
                    params['metric'] = hp.choice("metric",['euclidean', 'manhattan'])
                    params["p"] = hp.choice("p", [1, 2])


                    def objective(params):
                        model = KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]),
                                                   leaf_size=int(params["leaf_size"]),
                                                   metric=params['metric'],
                                                   p=params["p"])
                        accuracy = cross_val_score(model, X_train, y_train, cv=fold).mean()

                        # We aim to maximize accuracy, therefore we return it as a negative value
                        return {'loss': -accuracy, 'status': STATUS_OK}


                    met = {0 : 'euclidean', 1 : 'manhattan'}
                    pv = {0 :  1, 1 : 2}
                    trials = Trials()
                    best = fmin(fn=objective,
                                space=params,
                                algo=tpe.suggest,
                                max_evals=80,
                                trials=trials)
                    clf = KNeighborsClassifier(n_neighbors=int(best["n_neighbors"]),
                                                   leaf_size=int(best["leaf_size"]),
                                                   metric=met[best['metric']],
                                                   p=pv[best["p"]]).fit(X_train, y_train)

                    y_pred = clf.predict(X_test)
                    confusion_matrix(y_test, y_pred)
                    accuracy_score(y_test, y_pred)
                    classification_report(y_test, y_pred)

                    st.code("confusion_matrix {} {}".format(hyperop,confusion_matrix(y_test, y_pred)))
                    st.code(
                        "Classification report {} {}".format(hyperop,classification_report(y_test, y_pred)))
                    st.markdown(f"""
                                                                    <div style="background:	#757575 ;padding:1px">
                                                                    <h3 style="color:white;text-align:left;">Accuracy Score {hyperop} {accuracy_score(y_test, y_pred)}</h3>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)

                    st.sidebar.markdown(f"""
                                            <div style="background:#025246 ;padding:1px">
                                            <h3 style="color:white;text-align:left;">{hyperop} </h3>
                                            </div>
                                            """, unsafe_allow_html=True)
                    st.sidebar.markdown(f"""
                                                                <div style="background:#025246 ;padding:1px">
                                                                <h3 style="color:white;text-align:left;">Acc : {accuracy_score(y_test, y_pred)}</h3>
                                                                </div>
                                                                """, unsafe_allow_html=True)

                    st.code({f"""#Model {clf}"""})
                    st.write(f"### Code for {hyperop} and {models_name}")

                    st.code(f"""params = dict()
    if models_name == "KNN":
    params = dict()
    params["n_neighbors"] = hp.uniform("n_neighbors", 2, 50)
    params["leaf_size"] = hp.uniform("leaf_size", 1, 50)
    params['metric'] = hp.choice("metric",['euclidean', 'manhattan'])
    params["p"] = hp.choice("p", [1, 2])
    def objective(params):
        model = KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]),
                                    leaf_size=int(params["leaf_size"]),
                                    metric=params['metric'],
                                    p=params["p"])
        accuracy = cross_val_score(model, X_train, y_train, cv={fold}).mean()
        # We aim to maximize accuracy, therefore we return it as a negative value
        return {{'loss': -accuracy, 'status': STATUS_OK}}
    met = {{0 : 'euclidean', 1 : 'manhattan'}}
    pv = {{0 :  1, 1 : 2}}
    trials = Trials()
    best = fmin(fn=objective,
                space=params,
                algo=tpe.suggest,
                max_evals=80,
                trials=trials)
    clf = KNeighborsClassifier(n_neighbors=int(best["n_neighbors"]),
                                leaf_size=int(best["leaf_size"]),
                                metric=met[best['metric']],
                                p=pv[best["p"]]).fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)
    accuracy_score(y_test, y_pred)
    classification_report(y_test, y_pred)""")
                elif models_name == "Logistic Regression":
                    params = dict()
                    params["C"] = hp.uniform('C', 0.01, 1000)
                    params["max_iter"] = hp.uniform('max_iter', 5, 500)
                    params["solver"] = hp.choice("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                    params["penalty"] = hp.choice("penalty",["l1","l2"])


                    def objective(params):
                        model = LogisticRegression( C = params["C"],
                                                    max_iter = int(params["max_iter"]),
                                                    solver = params["solver"],
                                                    penalty = params["penalty"])
                        accuracy = cross_val_score(model, X_train, y_train, cv=fold).mean()

                        # We aim to maximize accuracy, therefore we return it as a negative value
                        return {'loss': -accuracy, 'status': STATUS_OK}


                    sol = {0: 'newton-cg', 1: 'lbfgs', 2: 'liblinear', 3: 'sag', 4: 'saga'}
                    pen = {0: 'l1', 1: 'l2'}
                    trials = Trials()
                    best = fmin(fn=objective,
                                space=params,
                                algo=tpe.suggest,
                                max_evals=80,
                                trials=trials)
                    clf = LogisticRegression(C=best["C"],
                                             max_iter = int(best["max_iter"]),
                                             solver = sol[best["solver"]],
                                             penalty=pen[best["penalty"]]).fit(X_train, y_train)

                    y_pred = clf.predict(X_test)
                    confusion_matrix(y_test, y_pred)
                    accuracy_score(y_test, y_pred)
                    classification_report(y_test, y_pred)
                    st.code("confusion_matrix {} {}".format(hyperop, confusion_matrix(y_test, y_pred)))
                    st.code(
                        "Classification report {} {}".format(hyperop, classification_report(y_test, y_pred)))
                    st.markdown(f"""<div style="background:	#757575 ;padding:1px"><h3 style="color:white;text-align:left;">Accuracy Score {hyperop} {accuracy_score(y_test, y_pred)}</h3></div>""", unsafe_allow_html=True)
                    st.sidebar.markdown(f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">{hyperop} </h3></div>""", unsafe_allow_html=True)
                    st.sidebar.markdown(f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">Acc : {accuracy_score(y_test, y_pred)}</h3></div>""", unsafe_allow_html=True)
                    st.code({f"""#Model {clf}"""})
                    st.write(f"### Code for {hyperop} and {models_name}")
                    st.code(f"""params=dict()
    params["C"] = hp.uniform('C', 0.01, 1000)
    params["max_iter"] = hp.uniform('max_iter', 5, 500)
    params["solver"] = hp.choice("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    params["penalty"] = hp.choice("penalty",["l1","l2"])
    def objective(params):
        model = LogisticRegression( C = params["C"],
                                    max_iter = int(params["max_iter"]),
                                    solver = params["solver"],
                                    penalty = params["penalty"])
        accuracy = cross_val_score(model, X_train, y_train, cv={fold}).mean()
        # We aim to maximize accuracy, therefore we return it as a negative value
        return {{'loss': -accuracy, 'status': STATUS_OK}}
    sol = {{0: 'newton-cg', 1: 'lbfgs', 2: 'liblinear', 3: 'sag', 4: 'saga'}}
    pen = {{0: 'l1', 1: 'l2'}}
    trials = Trials()
    best = fmin(fn=objective,
                space=params,
                algo=tpe.suggest,
                max_evals=80,
                trials=trials)
    clf = LogisticRegression(C=best["C"],
                            max_iter = int(best["max_iter"]),
                            solver = sol[best["solver"]],
                            penalty=pen[best["penalty"]]).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)
    accuracy_score(y_test, y_pred)
    classification_report(y_test, y_pred)""")

                elif models_name == "SVM":
                    params = dict()
                    params["C"] = hp.uniform("C", 0.01, 1000.0)
                    params["gamma"] = hp.uniform("gamma", 0.01, 1000.0)

                    def objective(params):
                        model = SVC(C = params["C"],
                                    gamma = params["gamma"])
                        accuracy = cross_val_score(model, X_train, y_train, cv=fold).mean()

                        # We aim to maximize accuracy, therefore we return it as a negative value
                        return {'loss': -accuracy, 'status': STATUS_OK}


                    trials = Trials()
                    best = fmin(fn=objective,
                                space=params,
                                algo=tpe.suggest,
                                max_evals=80,
                                trials=trials)
                    clf = SVC(C=best["C"],
                              gamma=best["gamma"]).fit(X_train,y_train)

                    y_pred = clf.predict(X_test)
                    confusion_matrix(y_test, y_pred)
                    accuracy_score(y_test, y_pred)
                    classification_report(y_test, y_pred)
                    st.code("confusion_matrix {} {}".format(hyperop, confusion_matrix(y_test, y_pred)))
                    st.code(
                        "Classification report {} {}".format(hyperop, classification_report(y_test, y_pred)))
                    st.markdown(
                        f"""<div style="background:	#757575 ;padding:1px"><h3 style="color:white;text-align:left;">Accuracy Score {hyperop} {accuracy_score(y_test, y_pred)}</h3></div>""",
                        unsafe_allow_html=True)
                    st.sidebar.markdown(
                        f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">{hyperop} </h3></div>""",
                        unsafe_allow_html=True)
                    st.sidebar.markdown(
                        f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">Acc : {accuracy_score(y_test, y_pred)}</h3></div>""",
                        unsafe_allow_html=True)
                    st.code({f"""#Model {clf}"""})
                    st.write(f"### Code for {hyperop} and {models_name}")
                    st.code(f"""params = dict()
    params["C"] = hp.uniform("C", 0.01, 1000.0)
    params["gamma"] = hp.uniform("gamma", 0.01, 1000.0)
    def objective(params):
        model = SVC(C = params["C"],
                gamma = params["gamma"])
        accuracy = cross_val_score(model, X_train, y_train, cv={fold}).mean()
        # We aim to maximize accuracy, therefore we return it as a negative value
        return {{'loss': -accuracy, 'status': STATUS_OK}}
    trials = Trials()
    best = fmin(fn=objective,
                space=params,
                algo=tpe.suggest,
                max_evals=80,
                trials=trials)
    clf = SVC(C=best["C"],
              gamma=best["gamma"]).fit(X_train,y_train)
                    
    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)
    accuracy_score(y_test, y_pred)
    classification_report(y_test, y_pred)""")
                elif models_name == "XGBoost":
                    params = dict()
                    params["max_depth"] = hp.quniform('max_depth', 10, 1200, 10)
                    params["n_estimators"] = hp.uniform("n_estimators", 20, 2000)
                    params["min_child_weight"] = hp.uniform("min_child_weight", 1, 5)
                    params["colsample_bytree"] = hp.uniform("colsample_bytree", 0.1, 0.9)
                    params["learning_rate"] = hp.uniform("learning_rate", 0.001, 0.999)
                    params["subsample"] = hp.uniform("subsample", 0.01, 0.99)
                    boost = {0: 'gbdt', 1: 'dart', 2: 'goss'}


                    def objective(params):
                        model = XGBClassifier(max_depth=int(params["max_depth"]),
                                                   n_estimators=int(params["n_estimators"]),
                                                   colsample_bytree=params["colsample_bytree"],
                                                   min_child_weight=int(params["min_child_weight"]),
                                                   subsample=params["subsample"],
                                                   learning_rate=params["learning_rate"])
                        accuracy = cross_val_score(model, X_train, y_train, cv=fold).mean()

                        # We aim to maximize accuracy, therefore we return it as a negative value
                        return {'loss': -accuracy, 'status': STATUS_OK}


                    trials = Trials()
                    best = fmin(fn=objective,
                                space=params,
                                algo=tpe.suggest,
                                max_evals=80,
                                trials=trials)



                    clf = XGBClassifier(max_depth=int(best["max_depth"]),
                                             n_estimators=int(best["n_estimators"]),
                                             colsample_bytree=best["colsample_bytree"],
                                             min_child_weight=int(best["min_child_weight"]),
                                             subsample=best["subsample"],
                                             learning_rate=best["learning_rate"]).fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    confusion_matrix(y_test, y_pred)
                    accuracy_score(y_test, y_pred)
                    classification_report(y_test, y_pred)
                    st.code("confusion_matrix {} {}".format(hyperop, confusion_matrix(y_test, y_pred)))
                    st.code(
                        "Classification report {} {}".format(hyperop, classification_report(y_test, y_pred)))
                    st.markdown(
                        f"""<div style="background:	#757575 ;padding:1px"><h3 style="color:white;text-align:left;">Accuracy Score {hyperop} {accuracy_score(y_test, y_pred)}</h3></div>""",
                        unsafe_allow_html=True)
                    st.sidebar.markdown(
                        f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">{hyperop} </h3></div>""",
                        unsafe_allow_html=True)
                    st.sidebar.markdown(
                        f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">Acc : {accuracy_score(y_test, y_pred)}</h3></div>""",
                        unsafe_allow_html=True)
                    st.code({f"""#Model {clf}"""})
                    st.write(f"### Code for {hyperop} and {models_name}")
                    st.code(f"""params = dict()
    params["max_depth"] = hp.quniform('max_depth', 10, 1200, 10)
    params["n_estimators"] = hp.uniform("n_estimators", 20, 2000)
    params["min_child_weight"] = hp.uniform("min_child_weight", 1, 5)
    params["colsample_bytree"] = hp.uniform("colsample_bytree", 0.1, 0.9)
    params["learning_rate"] = hp.uniform("learning_rate", 0.001, 0.999)
    params["subsample"] = hp.uniform("subsample", 0.01, 0.99)
    def objective(params):
        model = XGBClassifier(max_depth=int(params["max_depth"]),
                              n_estimators=int(params["n_estimators"]),
                              colsample_bytree=params["colsample_bytree"],
                              min_child_weight=int(params["min_child_weight"]),
                              subsample=params["subsample"],
                             learning_rate=params["learning_rate"])
        accuracy = cross_val_score(model, X_train, y_train, cv={fold}).mean()
        # We aim to maximize accuracy, therefore we return it as a negative value
        return {{'loss': -accuracy, 'status': STATUS_OK}}
    trials = Trials()
    best = fmin(fn=objective,
                space=params,
                algo=tpe.suggest,
                max_evals=80,
                trials=trials)
    clf = XGBClassifier(max_depth=int(best["max_depth"]),
                        n_estimators=int(best["n_estimators"]),
                        colsample_bytree=best["colsample_bytree"],
                        min_child_weight=int(best["min_child_weight"]),
                        subsample=best["subsample"],
                        learning_rate=best["learning_rate"]).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)
    accuracy_score(y_test, y_pred)
    classification_report(y_test, y_pred)""")
                elif models_name == "LightGBM":
                    params=dict()
                    params["max_depth"] = hp.quniform('max_depth', 10, 1200, 10)
                    params["reg_alpha"] = hp.uniform("reg_alpha", 2, 20)
                    params["n_estimators"] = hp.quniform("n_estimators", 20, 2000,10)
                    params["boosting_type"] = hp.choice("boosting_type", ['gbdt', 'dart', 'goss'])
                    params["reg_lambda"] = hp.uniform("reg_lambda", 1, 20)
                    params["num_leaves"] = hp.uniform("num_leaves", 2, 300)
                    params["colsample_bytree"] = hp.uniform("colsample_bytree", 0.1, 0.9)
                    params["min_child_weight"] = hp.uniform("min_child_weight", 1, 5)
                    params["min_child_samples"] = hp.uniform("min_child_samples", 1, 5)
                    params["learning_rate"] = hp.uniform("learning_rate", 0.001, 0.999)
                    boost = {0: 'gbdt', 1: 'dart',2:'goss'}

                    def objective(params):
                        model = lgb.LGBMClassifier(max_depth = int(params["max_depth"]),
                                                   reg_alpha = params["reg_alpha"],
                                                   n_estimators = int(params["n_estimators"]),
                                                   boosting_type = params["boosting_type"],
                                                   reg_lambda = params["reg_lambda"],
                                                   num_leaves = int(params["num_leaves"]),
                                                   colsample_bytree = params["colsample_bytree"],
                                                   min_child_weight = int(params["min_child_weight"]),
                                                   min_child_samples = int(params["min_child_samples"]),
                                                   learning_rate = params["learning_rate"])
                        accuracy = cross_val_score(model, X_train, y_train, cv = fold).mean()

                        # We aim to maximize accuracy, therefore we return it as a negative value
                        return {'loss': -accuracy, 'status': STATUS_OK}


                    trials = Trials()
                    best = fmin(fn=objective,
                                space=params,
                                algo=tpe.suggest,
                                max_evals=80,
                                trials=trials)

                    boost = {0: 'gbdt', 1: 'dart', 2 : 'goss'}

                    clf = lgb.LGBMClassifier(max_depth = int(best["max_depth"]),
                                                   reg_alpha = best["reg_alpha"],
                                                   n_estimators = int(best["n_estimators"]),
                                                   boosting_type = boost[best["boosting_type"]],
                                                   reg_lambda  = best["reg_lambda"],
                                                   num_leaves= int(best["num_leaves"]),
                                                   colsample_bytree = best["colsample_bytree"],
                                                   min_child_weight = int(best["min_child_weight"]),
                                                   min_child_samples = int(best["min_child_samples"]),
                                                   learning_rate = best["learning_rate"]).fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    confusion_matrix(y_test, y_pred)
                    accuracy_score(y_test, y_pred)
                    classification_report(y_test, y_pred)
                    st.code("confusion_matrix {} {}".format(hyperop, confusion_matrix(y_test, y_pred)))
                    st.code(
                        "Classification report {} {}".format(hyperop, classification_report(y_test, y_pred)))
                    st.markdown(
                        f"""<div style="background:	#757575 ;padding:1px"><h3 style="color:white;text-align:left;">Accuracy Score {hyperop} {accuracy_score(y_test, y_pred)}</h3></div>""",
                        unsafe_allow_html=True)
                    st.sidebar.markdown(
                        f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">{hyperop} </h3></div>""",
                        unsafe_allow_html=True)
                    st.sidebar.markdown(
                        f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">Acc : {accuracy_score(y_test, y_pred)}</h3></div>""",
                        unsafe_allow_html=True)
                    st.code({f"""#Model {clf}"""})
                    st.write(f"### Code for {hyperop} and {models_name}")
                    st.code(f"""params=dict()
    params["max_depth"] = hp.quniform('max_depth', 10, 1200, 10)
    params["reg_alpha"] = hp.uniform("reg_alpha", 2, 20)
    params["n_estimators"] = hp.quniform("n_estimators", 20, 2000,10)
    params["boosting_type"] = hp.choice("boosting_type", ['gbdt', 'dart', 'goss'])
    params["reg_lambda"] = hp.uniform("reg_lambda", 1, 20)
    params["num_leaves"] = hp.uniform("num_leaves", 2, 300)
    params["colsample_bytree"] = hp.uniform("colsample_bytree", 0.1, 0.9)
    params["min_child_weight"] = hp.uniform("min_child_weight", 1, 5)
    params["min_child_samples"] = hp.uniform("min_child_samples", 1, 5)
    params["learning_rate"] = hp.uniform("learning_rate", 0.001, 0.999)
    boost = {{0: 'gbdt', 1: 'dart',2:'goss'}}
    
    def objective(params):
        model = lgb.LGBMClassifier(max_depth = int(params["max_depth"]),
                                    reg_alpha = params["reg_alpha"],
                                    n_estimators = int(params["n_estimators"]),
                                    boosting_type = params["boosting_type"],
                                    reg_lambda = params["reg_lambda"],
                                    num_leaves = int(params["num_leaves"]),
                                    colsample_bytree = params["colsample_bytree"],
                                    min_child_weight = int(params["min_child_weight"]),
                                    min_child_samples = int(params["min_child_samples"]),
                                    learning_rate = params["learning_rate"])
        accuracy = cross_val_score(model, X_train, y_train, cv = {fold}).mean()
    
        # We aim to maximize accuracy, therefore we return it as a negative value
        return {{'loss': -accuracy, 'status': STATUS_OK}}
    
    
    trials = Trials()
    best = fmin(fn=objective,
                space=params,
                algo=tpe.suggest,
                max_evals=80,
                trials=trials)
    
    boost = {{0: 'gbdt', 1: 'dart', 2 : 'goss'}}
    
    clf = lgb.LGBMClassifier(max_depth = int(best["max_depth"]),
                            reg_alpha = best["reg_alpha"],
                            n_estimators = int(best["n_estimators"]),
                            boosting_type = boost[best["boosting_type"]],
                            reg_lambda  = best["reg_lambda"],
                            num_leaves= int(best["num_leaves"]),
                            colsample_bytree = best["colsample_bytree"],
                            min_child_weight = int(best["min_child_weight"]),
                            min_child_samples = int(best["min_child_samples"]),
                            learning_rate = best["learning_rate"]).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)
    accuracy_score(y_test, y_pred)
    classification_report(y_test, y_pred) """)


                else:

                    params = dict()
                    params["n_estimators"] = hp.choice('n_estimators',[10,50,300,750,1200,1300,1500])
                    params["criterion"] = hp.choice('criterion', ['entropy', 'gini'])
                    params["max_features"] = hp.choice('max_features', ['auto', 'sqrt','log2', None])
                    params["min_samples_leaf"] = hp.uniform('min_samples_leaf', 0,0.5)
                    params["min_samples_split"] = hp.uniform ('min_samples_split', 0,0.5)
                    params["max_depth"] = hp.quniform('max_depth', 2, 1200, 10)
                    clf_tune = RandomForestClassifier()


                    def objective(params):
                        model = RandomForestClassifier(criterion=params['criterion'],
                                                       max_depth=params['max_depth'],
                                                       max_features=params['max_features'],
                                                       min_samples_leaf=params['min_samples_leaf'],
                                                       min_samples_split=params['min_samples_split'],
                                                       n_estimators=params['n_estimators'],
                                                       )
                        accuracy = cross_val_score(model,X_train, y_train, cv = fold).mean()

                        # We aim to maximize accuracy, therefore we return it as a negative value
                        return {'loss': -accuracy, 'status': STATUS_OK}


                    trials = Trials()
                    best = fmin(fn=objective,
                                space=params,
                                algo=tpe.suggest,
                                max_evals=80,
                                trials=trials)

                    crit = {0: 'entropy', 1: 'gini'}
                    feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
                    est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200, 5: 1300, 6: 1500}

                    clf = RandomForestClassifier(criterion=crit[best['criterion']], max_depth=best['max_depth'],
                                                           max_features=feat[best['max_features']],
                                                           min_samples_leaf=best['min_samples_leaf'],
                                                           min_samples_split=best['min_samples_split'],
                                                           n_estimators=est[best['n_estimators']]).fit(X_train,y_train)
                    y_pred = clf.predict(X_test)
                    confusion_matrix(y_test, y_pred)
                    accuracy_score(y_test, y_pred)
                    classification_report(y_test, y_pred)
                    st.code("confusion_matrix {} {}".format(hyperop, confusion_matrix(y_test, y_pred)))
                    st.code(
                        "Classification report {} {}".format(hyperop, classification_report(y_test, y_pred)))
                    st.markdown(
                        f"""<div style="background:	#757575 ;padding:1px"><h3 style="color:white;text-align:left;">Accuracy Score {hyperop} {accuracy_score(y_test, y_pred)}</h3></div>""",
                        unsafe_allow_html=True)
                    st.sidebar.markdown(
                        f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">{hyperop} </h3></div>""",
                        unsafe_allow_html=True)
                    st.sidebar.markdown(
                        f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">Acc : {accuracy_score(y_test, y_pred)}</h3></div>""",
                        unsafe_allow_html=True)
                    st.code({f"""#Model {clf}"""})
                    st.write(f"### Code for {hyperop} and {models_name}")
                    st.code("""
        params = dict()
        params["n_estimators"] = hp.choice('n_estimators',[10,50,300,750,1200,1300,1500])
        params["criterion"] = hp.choice('criterion', ['entropy', 'gini'])
        params["max_features"] = hp.choice('max_features', ['auto', 'sqrt','log2', None])
        params["min_samples_leaf"] = hp.uniform('min_samples_leaf', 0, 0.5)
        params["min_samples_split"] = hp.uniform ('min_samples_split', 0, 1)
        params["max_depth"] = hp.quniform('max_depth', 10, 1200, 10)
    
    
    
        def objective(clf_tune,params):
            accuracy = cross_val_score(clf_tune,df, y,fit_params=params, cv = fold).mean()
            # We aim to maximize accuracy, therefore we return it as a negative value
            return {'loss': -accuracy, 'status': STATUS_OK}
        trials = Trials()
        best = fmin(fn=objective,
                    space=params,
                    algo=tpe.suggest,
                    max_evals=80,
                    trials=trials)
    
        crit = {0: 'entropy', 1: 'gini'}
        feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
        est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200, 5: 1300, 6: 1500}
    
        clf = RandomForestClassifier(criterion=crit[best['criterion']], max_depth=best['max_depth'],
                                    max_features=feat[best['max_features']],
                                    min_samples_leaf=best['min_samples_leaf'],
                                    min_samples_split=best['min_samples_split'],
                                    n_estimators=est[best['n_estimators']]).fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        confusion_matrix(y_test, y_pred)
        accuracy_score(y_test, y_pred)
        classification_report(y_test, y_pred)""")

        if hyperop=="Genetic Algorithms (TPOT Classifier)":
            params = dict()

            if models_name == "KNN":
                K = [int(x) for x in np.linspace(start=3, stop=20, num=10)]
                params["n_neighbors"] = K
                leaf_size = [int(x) for x in np.linspace(start=1, stop=50, num=10)]
                params["leaf_size"] = leaf_size
                p = [1, 2]
                params['metric'] = ['euclidean', 'manhattan']
                params["p"] = p
                clf_tune = "sklearn.neighbors.KNeighborsClassifier"
            elif models_name == "Logistic Regression":

                C = [int(x) for x in np.linspace(start=0.001, stop=1000.0, num=10)]
                penalty = ['l2', 'l1']
                max_iter = [int(x) for x in np.linspace(start=1, stop=500, num=10)]
                solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                params["C"] = C
                params["penalty"] = penalty
                params["max_iter"] = max_iter
                params["solver"] = solver
                iteration = 200
                clf_tune = "sklearn.linear_model.LogisticRegression"
            elif models_name == "SVM":

                C = [int(x) for x in np.linspace(start=0.001, stop=1000.0, num=10)]
                params["C"] = C
                gamma_range = [int(x) for x in np.linspace(start=0.01, stop=1000.0, num=10)]
                params["gamma"] = gamma_range
                iteration = 100
                clf_tune = "sklearn.svm.SVC"
            elif models_name == "XGBoost":

                max_depth = [int(x) for x in np.linspace(start=2, stop=200, num=10)]
                n_estimators = [int(x) for x in np.linspace(start=50, stop=2000, num=10)]
                subsample = [x for x in np.linspace(start=0.01, stop=1.0, num=4)]
                learning_rate = [x for x in np.linspace(start=0.01, stop=0.90, num=4)]
                colsample_bytree = [x for x in np.linspace(start=0.01, stop=0.90, num=4)]
                min_child_weight = [int(x) for x in np.linspace(start=1, stop=5, num=4)]
                params["max_depth"] = max_depth
                params["n_estimators"] = n_estimators
                params["min_child_weight"] = min_child_weight
                params["colsample_bytree"] = colsample_bytree
                params["learning_rate"] = learning_rate
                params["subsample"] = subsample
                iteration = 200
                clf_tune = "xgboost.XGBClassifier"
            elif models_name == "LightGBM":

                max_depth = [int(x) for x in np.linspace(start=-1, stop=200, num=5)]
                reg_alpha = [x for x in np.linspace(start=0.0, stop=20.0, num=4)]
                n_estimators = [int(x) for x in np.linspace(start=50, stop=2000, num=10)]
                boosting_type = ['gbdt', 'dart', 'goss']
                reg_lambda = [x for x in np.linspace(start=0.001, stop=20.0, num=5)]
                num_leaves = [int(x) for x in np.linspace(start=2, stop=300, num=10)]
                colsample_bytree = [x for x in np.linspace(start=0.1, stop=0.9, num=5)]
                min_child_samples = [int(x) for x in np.linspace(start=1, stop=5, num=3)]
                min_child_weight = [x for x in np.linspace(start=0.001, stop=1.0, num=5)]
                learning_rate = [x for x in np.linspace(start=0.001, stop=0.999, num=5)]
                params["max_depth"] = max_depth
                params["reg_alpha"] = reg_alpha
                params["n_estimators"] = n_estimators
                params["boosting_type"] = boosting_type
                params["reg_lambda"] = reg_lambda
                params["num_leaves"] = num_leaves
                params["colsample_bytree"] = colsample_bytree
                params["min_child_weight"] = min_child_weight
                params["min_child_samples"] = min_child_samples
                params["learning_rate"] = learning_rate
                iteration = 200
                clf_tune = "lightgbm.LGBMClassifier"

            else:

                params["n_estimators"] = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
                params["criterion"] = ['entropy', 'gini']
                params["max_features"] = ['auto', 'sqrt', 'log2']
                params["min_samples_leaf"] = [int(x) for x in np.linspace(start=1, stop=10, num=5)]
                params["min_samples_split"] = [int(x) for x in np.linspace(start=2, stop=15, num=3)]
                params["max_depth"] = [int(x) for x in np.linspace(start=1, stop=1000, num=10)]
                clf_tune = 'sklearn.ensemble.RandomForestClassifier'
                iteration = 400
            clf = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                             verbosity=2, early_stop=12,
                                             config_dict={ clf_tune : params},
                                             cv=fold, scoring='accuracy')
            clf.fit(X_train, y_train)
            st.code("confusion_matrix Bayesian Optimization (Hyperopt): {}".format(confusion_matrix(y_test, y_pred)))
            st.code("Accuracy Score Bayesian Optimization (Hyperopt) {}".format(accuracy_score(y_test, y_pred)))
            st.sidebar.write("Accuracy Score Tuned Genetic Algorithms (TPOT Classifier){}".format(clf.score(X_test, y_test)))
            st.code(
                "Classification report Bayesian Optimization (Hyperopt): {}".format(classification_report(y_test, y_pred)))
            st.code(f"""#{models_name}
    tpot_classifier = TPOTClassifier(generations=5, 
                                population_size=24, 
                                offspring_size=12,
                                verbosity=2, 
                                early_stop=12,
                                config_dict={{ {clf_tune} : params}},
                                cv={fold}, 
                                scoring='accuracy')
    tpot_classifier.fit(X_train, y_train)
    accuracy = tpot_classifier.score(X_test, y_test)""")
        if  hyperop == "Optuna- Automate Hyperparameter Tuning":
            params = dict()
            if models_name == "KNN":
                def objective(trial):
                    params = dict()
                    params["n_neighbors"] = trial.suggest_int("n_neighbors", 2, 50)
                    params["leaf_size"] = trial.suggest_int("leaf_size", 1, 50)
                    params['metric'] = trial.suggest_categorical("metric", ['euclidean', 'manhattan'])
                    params["p"] = trial.suggest_categorical("p", [1, 2])
                    model = KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]),
                                                 leaf_size=int(params["leaf_size"]),
                                                 metric=params['metric'],
                                                 p=params["p"])
                    return cross_val_score(model, X_train, y_train, cv=fold,n_jobs = -1).mean()
                st.write(f"Code for Optima ({models_name})")
                st.code(f"""def objective(trial):
        params = dict()
        params["n_neighbors"] = trial.suggest_int("n_neighbors", 2, 50)
        params["leaf_size"] = trial.suggest_int("leaf_size", 1, 50)
        params['metric'] = trial.suggest_categorical("metric", ['euclidean', 'manhattan'])
        params["p"] = trial.suggest_categorical("p", [1, 2])
        model = KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]),
                                     leaf_size=int(params["leaf_size"]),
                                     metric=params['metric'],
                                     p=params["p"])
        return cross_val_score(model, X_train, y_train, cv=fold,n_jobs = -1).mean()""")
            elif models_name == "Logistic Regression":
                def objective(trial):
                    params = dict()
                    params["C"] = trial.suggest_float('C', 0.01, 1000)
                    params["max_iter"] = trial.suggest_int('max_iter', 5, 500)
                    params["solver"] = trial.suggest_categorical("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                    params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
                    clf = LogisticRegression(C=params["C"],
                                               max_iter=int(params["max_iter"]),
                                               solver=params["solver"],
                                               penalty=params["penalty"])
                    return cross_val_score(clf, X_train, y_train,n_jobs =-1, cv=fold).mean()


                st.write(f"Code for Optima ({models_name})")
                st.code(f"""def objective(trial):
        params = dict()
        params["C"] = trial.suggest_float('C', 0.01, 1000)
        params["max_iter"] = trial.suggest_int('max_iter', 5, 500)
        params["solver"] = trial.suggest_categorical("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        clf = LogisticRegression(C=params["C"],
                                 max_iter=int(params["max_iter"]),
                                 solver=params["solver"],
                                 penalty=params["penalty"])
        return cross_val_score(clf, X_train, y_train,n_jobs =-1, cv={fold}).mean()""")

            elif models_name == "SVM":
                def objective(trial):
                    params = dict()
                    params["C"] = trial.suggest_float("C", 0.01, 1000.0)
                    params["gamma"] = trial.suggest_float("gamma", 0.01, 1000.0)
                    clf = SVC(C=params["C"],
                              gamma=params["gamma"])
                    return cross_val_score(clf,X_train,y_train,cv=fold,n_jobs=-1).mean()
                st.write(f"Code for Optima ({models_name})")
                st.code(f"""def objective(trial):
        params = dict()
        params["C"] = trial.suggest_float("C", 0.01, 1000.0)
        params["gamma"] = trial.suggest_float("gamma", 0.01, 1000.0)
        clf = SVC(C=params["C"],
                    gamma=params["gamma"])
        return cross_val_score(clf,X_train,y_train,cv={fold},n_jobs=-1).mean()""")



            elif models_name == "XGBoost":
                def objective(trial):
                    params = dict()
                    params["max_depth"] = trial.suggest_int('max_depth', 10, 1200, 10)
                    params["n_estimators"] = trial.suggest_int("n_estimators", 20, 2000)
                    params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 5)
                    params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.1, 0.9)
                    params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 0.999)
                    params["subsample"] = trial.suggest_float("subsample", 0.01, 0.99)

                    clf = XGBClassifier(max_depth=(params["max_depth"]),
                                          n_estimators=(params["n_estimators"]),
                                          colsample_bytree=params["colsample_bytree"],
                                          min_child_weight=(params["min_child_weight"]),
                                          subsample=params["subsample"],
                                          learning_rate=params["learning_rate"])
                    return cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=fold).mean()
                st.write(f"Code for Optima ({models_name})")
                st.code(f"""def objective(trial):
        params = dict()
        params["max_depth"] = trial.suggest_int('max_depth', 10, 1200, 10)
        params["n_estimators"] = trial.suggest_int("n_estimators", 20, 2000)
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 5)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.1, 0.9)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 0.999)
        params["subsample"] = trial.suggest_float("subsample", 0.01, 0.99)
    
        clf = XGBClassifier(max_depth=(params["max_depth"]),
                            n_estimators=(params["n_estimators"]),
                            colsample_bytree=params["colsample_bytree"],
                            min_child_weight=(params["min_child_weight"]),
                            subsample=params["subsample"],
                            learning_rate=params["learning_rate"])
        return cross_val_score(clf, X_train, y_train, n_jobs=-1, cv={fold}).mean()""")
            elif models_name == "LightGBM":
                def objective(trial):
                    params = dict()
                    params["max_depth"] = trial.suggest_int('max_depth', 10, 1200, 10)
                    params["reg_alpha"] = trial.suggest_int("reg_alpha", 2, 20)
                    params["n_estimators"] = trial.suggest_int("n_estimators", 20, 2000, 10)
                    params["boosting_type"] = trial.suggest_categorical("boosting_type", ['gbdt', 'dart', 'goss'])
                    params["reg_lambda"] = trial.suggest_int("reg_lambda", 1, 20,5)
                    params["num_leaves"] = trial.suggest_int("num_leaves", 2, 300,5)
                    params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.1, 0.9)
                    params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 5)
                    params["min_child_samples"] = trial.suggest_int("min_child_samples", 1, 5)
                    params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 0.999)
                    clf = lgb.LGBMClassifier(max_depth = params["max_depth"],
                                               reg_alpha = params["reg_alpha"],
                                               n_estimators = params["n_estimators"],
                                               boosting_type = params["boosting_type"],
                                               reg_lambda  =params["reg_lambda"],
                                               num_leaves = params["num_leaves"],
                                               colsample_bytree = params["colsample_bytree"],
                                               min_child_weight = params["min_child_weight"],
                                               min_child_samples = params["min_child_samples"],
                                               learning_rate = params["learning_rate"])
                    return cross_val_score(clf,X_train,y_train, n_jobs=-1, cv=fold).mean()
                st.write(f"Code for Optima ({models_name})")
                st.code(f"""def objective(trial):
        params = dict()
        params["max_depth"] = trial.suggest_int('max_depth', 10, 1200, 10)
        params["reg_alpha"] = trial.suggest_int("reg_alpha", 2, 20)
        params["n_estimators"] = trial.suggest_int("n_estimators", 20, 2000, 10)
        params["boosting_type"] = trial.suggest_categorical("boosting_type", ['gbdt', 'dart', 'goss'])
        params["reg_lambda"] = trial.suggest_int("reg_lambda", 1, 20,5)
        params["num_leaves"] = trial.suggest_int("num_leaves", 2, 300,5)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.1, 0.9)
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 5)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 1, 5)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 0.999)
        clf = lgb.LGBMClassifier(max_depth = params["max_depth"],
                                reg_alpha = params["reg_alpha"],
                                n_estimators = params["n_estimators"],
                                boosting_type = params["boosting_type"],
                                reg_lambda  =params["reg_lambda"],
                                num_leaves = params["num_leaves"],
                                colsample_bytree = params["colsample_bytree"],
                                min_child_weight = params["min_child_weight"],
                                min_child_samples = params["min_child_samples"],
                                learning_rate = params["learning_rate"])
        return cross_val_score(clf,X_train,y_train, n_jobs=-1, cv={fold}).mean()""")
            else:
                def objective(trial):
                    params = dict()
                    params["n_estimators"] = trial.suggest_int('n_estimators', 10,1500,10)
                    params["criterion"] = trial.suggest_categorical('criterion', ['entropy', 'gini'])
                    params["max_features"] = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
                    params["min_samples_leaf"] = trial.suggest_int('min_samples_leaf', 1, 10)
                    params["min_samples_split"] = trial.suggest_int('min_samples_split', 2, 20)
                    params["max_depth"] = trial.suggest_int('max_depth', 2, 1200, 10)

                    clf = RandomForestClassifier(criterion=params['criterion'],
                                                   max_depth=params['max_depth'],
                                                   max_features=params['max_features'],
                                                   min_samples_leaf=params['min_samples_leaf'],
                                                   min_samples_split=params['min_samples_split'],
                                                   n_estimators=params['n_estimators'])
                    return cross_val_score(clf,X_train,y_train, n_jobs=-1, cv=fold).mean()
                st.code(f"""def objective(trial):
    params = dict()
    params["n_estimators"] = trial.suggest_int('n_estimators', 10,1500,10)
    params["criterion"] = trial.suggest_categorical('criterion', ['entropy', 'gini'])
    params["max_features"] = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
    params["min_samples_leaf"] = trial.suggest_int('min_samples_leaf', 1, 10)
    params["min_samples_split"] = trial.suggest_int('min_samples_split', 2, 20)
    params["max_depth"] = trial.suggest_int('max_depth', 2, 1200, 10)

    clf = RandomForestClassifier(criterion=params['criterion'],
                            max_depth=params['max_depth'],
                            max_features=params['max_features'],
                            min_samples_leaf=params['min_samples_leaf'],
                            min_samples_split=params['min_samples_split'],
                            n_estimators=params['n_estimators'])
    return cross_val_score(clf,X_train,y_train, n_jobs=-1, cv={fold}).mean()""")


            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            trial = study.best_trial
            trial.params
            st.code("Best hyperparameters: {}".format(trial.params))

            st.markdown(
                f"""<div style="background:	#757575 ;padding:1px"><h3 style="color:white;text-align:left;">Accuracy Score {hyperop} {trial.value}</h3></div>""",
                unsafe_allow_html=True)
            st.sidebar.markdown(
                f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">{hyperop} </h3></div>""",
                unsafe_allow_html=True)
            st.sidebar.markdown(
                f"""<div style="background:#025246 ;padding:1px"><h3 style="color:white;text-align:left;">Acc : {trial.value}</h3></div>""", unsafe_allow_html=True)

            st.write(f"### Code for {hyperop} and {models_name}")
            st.code(f"""study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
trial = study.best_trial
trial.params""")





    else:
        st.write("You do not apply any Hyper Parameter Optimization")
        st.markdown(f"""You can apply these optimizers.
    
            - RandomizedSearchCV-GridSearchCV (sequentially)
            
            - Bayesian Optimization (Hyperopt)
             
            - Optuna- Automate Hyperparameter Tuning
            
            - Genetic Algorithms (TPOT Classifier) """)
    st.cache()

    st.write(""" ## Download Model """)
    st.cache()
    def download_model(model):
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model .pkl File</a> (right-click and save as &lt;model&gt;.pkl)'
        st.markdown(href, unsafe_allow_html=True)
    try:
        download_model(clf)
    except:
        pass
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

    os.system('rm -f ./*.pkl')
    def pickle_all(key,value):
         pickle_out = open(key+".pkl", "wb")
         pickle.dump(value, pickle_out)
         pickle_out.close()
    try:
        pickle_all("df_test", df_test)
    except:
        pass
    try:
        pickle_all("target_column", target_column)
    except:
        pass
    try:
        pickle_all("df__test", df__test)
    except:
        pass
    try:
        pickle_all("X_Selected", X_Selected)
    except:
        pass
    try:
        pickle_all("clf", clf)
    except:
        pass
    try:
       pickle_all("models_name", models_name)
    except:
        pass
    try:
        pickle_all("drop_list", drop_list)
    except:
        pass
    try:
        pickle_all("corCol", corCol)
    except:
        pass
    try:
        pickle_all("encode_list", encode_list)
    except:
        pass
    try:
        pickle_all("Outliers_handle",Outliers_handle)
    except:
        pass
    try:
        pickle_all("numeric_cols", numeric_cols)
    except:
        pass
    try:
        pickle_all("col_after_endoded_all",col_after_endoded_all)
    except:
        pass
    try:
        pickle_all("Feature_importance", Feature_importance)
    except:
        pass
    try:
        pickle_all("standard_apply", standard_apply)
    except:
        pass
    try:
        pickle_all("scaler_obj", scaler_obj)
    except:
        pass
    try:
        pickle_all("g_cols", g_cols)
    except:
        pass
    try:
        pickle_all("scale_s", scale_s)
    except:
        pass
    try:
        pickle_all("use_pca", use_pca)
    except:
        pass
    try:
        pickle_all("pca1", pca1)
    except:
        pass
    try:
        pickle_all("s_cols", s_cols)
    except:
        pass







if PAGE=="Prediction and Testing":


    st.markdown(f"""
                                            <div style="background:#025246 ;padding:10px">
                                            <h3 style="color:white;text-align:left;"> Prediction and Test</h3>
                                            </div>
                                            """, unsafe_allow_html=True)





    try:
        pickle_in = open('df_test.pkl', 'rb')
        df_test = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('target_column.pkl', 'rb')
        target_column = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('df__test.pkl', 'rb')
        df__test = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('X_Selected.pkl', 'rb')
        X_Selected = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('clf.pkl', 'rb')
        clf= pickle.load(pickle_in)
    except:
        pass
    try:
       pickle_in = open('models_name.pkl', 'rb')
       models_name = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('drop_list.pkl', 'rb')
        drop_list = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('corCol.pkl', 'rb')
        corCol = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('encode_list.pkl', 'rb')
        encode_list = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('Outliers_handle.pkl', 'rb')
        Outliers_handle = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('numeric_cols .pkl', 'rb')
        numeric_cols = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('col_after_endoded_all.pkl', 'rb')
        col_after_endoded_all = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('Feature_importance.pkl', 'rb')
        Feature_importance = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('standard_apply.pkl', 'rb')
        standard_apply= pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('scaler_obj.pkl', 'rb')
        scaler_obj = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('g_cols.pkl', 'rb')
        g_cols = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('scale_s.pkl', 'rb')
        scale_s= pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('use_pca.pkl', 'rb')
        use_pca = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('pca1.pkl', 'rb')
        pca1 = pickle.load(pickle_in)
    except:
        pass
    try:
        pickle_in = open('s_cols.pkl', 'rb')
        s_cols = pickle.load(pickle_in)
    except:
        pass



    df_test.drop(target_column, axis=1, inplace=True)
    columns = df_test.columns

    cat_columns = df_test.select_dtypes(include=["object"]).columns



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
            dftest = dftest.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        except:
            pass
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

if PAGE == "Deployment":
    try:
        os.system('rm -f project_name.pkl')
    except:
        pass
    project_name =st.text_input("Write your project name")


    try:
        def pickle_all(key, value):
            pickle_out = open(key + ".pkl", "wb")
            pickle.dump(value, pickle_out)
            pickle_out.close()
        pickle_all("project_name", project_name)

        import marshal

        s = open("app_deploy.py").read()
        g = compile(s, '', 'exec')
        b = marshal.dumps(g)
        w = open('f-marshal.py', 'w')
        w.write('import marshal\n')
        w.write('exec(marshal.loads(' + repr(b) + '))')
        w.close()

        os.system('bash deploy_local.sh')
        try:
            with open("local_deployment.tar", "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'<a href="data:file/tar;base64,{b64}">Download File</a> (right-click and save as local_deployment.tar)'

                st.markdown(
                    f"""<div style="color:black";border-color:#F50057" class="box">Download the file to the desktop(right-click and save as local_deployment.tar).
                    Then type the commands below into git bash.</div>""",
                    unsafe_allow_html=True)

                st.markdown(href, unsafe_allow_html=True)
                st.code("tar -xf ~/Desktop/local_deployment.tar")
                st.code("cd ~/Desktop/local_deployment")
                st.markdown(f"""
                <!DOCTYPE html>
                <title>My Example</title>
                <style>
                    .box {{
                        background-color: transparent;
                        font-size: 1vw;
                        padding: 0vw;
                        margin: 1vw;
                        border: solid;
                        border-color:#F50057;
                    }}
                </style>
    
                <div class="box">In order to deploy to heroku, first of all you need an account.Open your command Prompt</div>""", unsafe_allow_html=True)

                st.markdown("https://devcenter.heroku.com/articles/heroku-cli  Download CLI ")

                st.code("heroku login")
                st.code("heroku create appname #write any name")





                st.code("bash ~/Desktop/local_deployment/local.sh")
        except:
            pass
    except:
        pass

















