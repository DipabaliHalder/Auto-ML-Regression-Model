import io
import pickle
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, pair_confusion_matrix, precision_score,r2_score, recall_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns

def info():
    st.subheader("What is this?")
    st.markdown("This is a simple AutoML regression app using Streamlit and Pandas Profiling. The application allows users to upload a dataset, perform exploratory data analysis (EDA) and then build and evaluate regression models.")
    st.subheader("How to use it?")
    st.markdown("**Dataset Overview:** It provides information about how the dataset looks and no. of null or duplicate values.")
    st.markdown("**Exploratory Data Analysis:** Analyze and Visualizing the dataset.")
    st.markdown("**Training & Evaluation:** Train and evaluate the model tab to create the model.pkl file.")
    st.markdown("**Predict:** Based on the model.pkl file created in 'Taining & Evaluation' tab make the predictions.")

def data_cleaning(df):
    #Display Dataframe
    st.write(f"**Here's the data from the uploaded file{df.shape}:**")
    st.dataframe(df.head())
    col1,col2=st.columns(2)
    with col1:
        # Display data.info()
        st.write("**DataFrame Information:**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    with col2:
        st.write("**Null Values:**")
        null_values = df.isnull().sum()
        st.write(null_values)

        # Check for duplicated rows
        duplicate_count = df.duplicated().sum()
        st.write(f"**Duplicated Rows: {duplicate_count}**")
    st.write("**Summary Statistics:**")
    st.write(df.describe())
    
def feature_imp(data):
    st.subheader("Feature Importance")
    st.write("Using permutation_importance()")
    # Check if there's a numeric target variable
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) > 0:
        target_column = st.selectbox("Select target variable for feature importance", numeric_columns)
        
        # Algorithm selection
        algorithm = st.selectbox("Choose the algorithm for feature importance:",
                                 ["Linear Regression", "Decision Tree", "Random Forest"])
        
        # Prepare the data
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train the selected model
        if algorithm == "Linear Regression":
            model = LinearRegression()
        elif algorithm == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        else:  # Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': perm_importance.importances_mean
        })
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        
        # Plot feature importances
        fig5 = px.bar(feature_importance_df, x='importance', y='feature', orientation='h',
                      labels={'importance': 'Importance', 'feature': 'Feature'},
                      color='importance',
                      color_continuous_scale='Viridis')
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.write("No numeric columns available for feature importance analysis.")
    
def auto_eda(data):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Column Distribution")
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        selected_column = st.selectbox("Select column for histogram", numeric_columns)
        fig1 = px.histogram(data, x=selected_column, color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Scatter Plot between Two Columns")
        scatter_x = st.selectbox("Select X-axis column", numeric_columns)
        scatter_y = st.selectbox("Select Y-axis column", numeric_columns, index=1)
        fig2 = px.scatter(data, x=scatter_x, y=scatter_y, color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig2, use_container_width=True)

    # Second row of visualizations
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Box Plot")
        selected_column_box = st.selectbox("Select column for box plot", numeric_columns)
        fig3 = px.box(data, y=selected_column_box, color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        st.subheader("Correlation Matrix")
        non_num_cols=[]
        for i in df.columns:
            if df[i].dtype == 'object':
                non_num_cols.append(i)
        d=df.drop(columns=non_num_cols)
        corr_matrix = d.corr()
        fig4 = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                         x=corr_matrix.columns,
                                         y=corr_matrix.columns,
                                         colorscale='Viridis'))
        st.plotly_chart(fig4, use_container_width=True)
    feature_imp(data)    


def train(data, target):
    features = st.multiselect("Select features for training:", 
                              [col for col in data.columns if col != target])
    
    if not features:
        st.warning("Please select at least one feature for training.")
        return

    model = st.selectbox("Choose the Algorithm:", ["Linear Regression", "Decision Tree", "Random Forest"], index=None)
    per = st.slider("Choose the percentage of data for Training:", min_value=10, max_value=100)
    
    if st.button("Train"):
        # Use only selected features and target
        X = data[features]
        y = data[target]

        label_encoders = {}
        for column in X.columns:
            if X[column].dtype == 'object':
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-per)/100, random_state=42)
        
        if model == "Linear Regression":
            model = LinearRegression()
        elif model == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model == "Random Forest":
            model = RandomForestRegressor()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        st.write(f"R2 score: {r2:.2f}")
        st.write(f"Mean Absolute Error: {mae:,.2f}")
        
        # Save model and selected features
        with open(f'{file}_model.pkl', 'wb') as f:
            pickle.dump((model, features, label_encoders), f)
        
        st.success("Model trained successfully!")
        
        # Plot Actual vs. Predicted values
        st.subheader("Actual vs. Predicted Prices")
        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid")
        errors = abs(y_test - y_pred)
        sns.scatterplot(x=y_test, y=y_pred, hue=errors, palette="coolwarm", size=errors, sizes=(20, 200), legend=False)
        sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal')
        plt.title(f"Model: {type(model).__name__}", fontsize=16)
        plt.xlabel(f"Actual {target}", fontsize=14)
        plt.ylabel(f"Predicted {target}", fontsize=14)
        st.pyplot(plt)
    

def predict(data, target):
    try:
        with open(f'{file}_model.pkl', 'rb') as f:
            model, selected_features, label_encoders = pickle.load(f)
        st.success("Model loaded successfully.")
    except:
        st.error("The model has not been trained yet. Please train the model in the 'Training & Evaluation' tab.")
        return

    input_data = {}

    for feature in selected_features:
        if data[feature].dtype == 'object':
            value = st.selectbox(f"{feature}", sorted(data[feature].unique()), index=None)
        else:
            value = st.number_input(f"{feature}")
        input_data[feature] = value

    if st.button('Predict'):
        input_df = pd.DataFrame([input_data])
        
        for column in label_encoders:
            if column in input_df.columns:
                input_df[column] = label_encoders[column].transform(input_df[column])

        prediction = model.predict(input_df)[0]
        st.subheader(f"{target.upper()}: {prediction:,.2f}")

st.header("Auto-ML Regression App")
uploaded_file = st.file_uploader("Upload your Dataset(.csv file)", type="csv")
target=None
try:
    file=str(uploaded_file.name).removesuffix(".csv")
except:
    st.markdown("No file uploaded.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    for i in df.columns:
        try:
            df[i]=pd.to_numeric(df[i])
        except:
            print(f"Failed to convert {i} to numeric value")

    tab1,tab2,tab3,tab4,tab5=st.tabs(["Home","Dataset Overview","Exploratory Data Analysis","Training & Evaluation","Predict"])
    with tab1:
        info()
    with tab2:
        data_cleaning(df)
    with tab3:
        auto_eda(df)
    with tab4:
        target = st.selectbox("Select Target:", sorted(df.columns), index=None)
        if target:
            train(df, target)
        else:
            st.warning("Please select a target variable.")

        #target=st.selectbox("Select Target:",sorted(df.columns),index=None)
        #train(df,target)
    with tab5:
        predict(df,target)