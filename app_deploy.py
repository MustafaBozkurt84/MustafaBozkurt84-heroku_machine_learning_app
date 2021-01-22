import pickle
pickle_in = open('project_name.pkl', 'rb')
project_name = pickle.load(pickle_in)

def prediction_deploy():
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
    import base64
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
    import os
    import streamlit.components.v1 as components
    import marshal, types
    st.markdown(f"""<div style=background:#025246 ;padding:10px"><h1 style="color:white;text-align:left;">{project_name}</h1></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#025246 ;padding:10px"><h3 style="color:white;text-align:left;"> Prediction and Test</h3></div>""", unsafe_allow_html=True)





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


    if True:
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
                results_table=result.iloc[:,0]
                st.write(results_table)


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
                        csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                    return f'<a href="data:file/csv;base64,{{b64}}" download="Output.csv">Download Output csv file</a>'

                st.markdown(get_table_download_link(output), unsafe_allow_html=True)
                st.cache()
                try:

                    def get_table_download_link_kaggle(df):

                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                        return f'<a href="data:file/csv;base64,{{b64}}" download="Submission.csv">Download Submission csv file for Kaggle </a>'

                    st.markdown(get_table_download_link_kaggle(outputkaggle), unsafe_allow_html=True)
                    st.write("https://github.com/MustafaBozkurt84")
                    st.write("https://streamlit-machine-learning-app.herokuapp.com/")
                except:
                    pass
    if st.button("Predict"):
                    Prediction(dftest, testfile)
                    st.write("https://github.com/MustafaBozkurt84")


prediction_deploy()




