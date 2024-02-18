import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import pandas as pd
import seaborn as sns

class App:
    def __init__(self):
        self.dataset_name = None
        self.classifier_name = None
        self.Init_Streamlit_Page()
        self.data = None
        self.params = dict()
        self.clf = None
        self.X, self.y = None, None
    
    def run(self):
        self.get_dataset()
        self.add_parameter_ui()
        if self.data is not None:
            self.generate()
    
    def Init_Streamlit_Page(self):
        st.title('Streamlit Example')

        st.write("""
        # Explore different classifier and datasets
        Which one is the best?
        """)

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Uploaded CSV', 'Breast Cancer Wisconsin',)
        )
        st.write(f"## {self.dataset_name} Dataset")

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Gaussian Naive Bayes')
        )
    
    def get_dataset(self):
        
        if self.dataset_name == 'Breast Cancer Wisconsin':
            self.data = self.load_breast_cancer()
        elif self.dataset_name == 'Uploaded CSV':
            uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                st.write("filename:", uploaded_file.name)
                self.data = self.load_from_csv(uploaded_file)
        else:
            st.write('No dataset selected')
            
        if self.data is not None:
            st.write('The first 10 rows of the dataset:')
            st.dataframe(self.data.head(10))
            st.write('The last 10 rows of the dataset with irrelevant columns removed:')
            if 'id' and 'Unnamed: 32' in self.data.columns:
                self.data.drop(columns=['id', 'Unnamed: 32'], axis=1, inplace=True)
            st.dataframe(self.data.tail(10))
            self.preprocess(self.data)
    
    def load_from_csv(self, uploaded_file):
        data = pd.read_csv(uploaded_file)
        return data

    def load_breast_cancer(self):
        data = pd.read_csv('data.csv')
        return data
    
    def preprocess(self,df):
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        self.X = df.drop('diagnosis', axis=1) # Features
        #print(f"X: {self.X.shape}")
        self.y = df['diagnosis'] # Target variable
        #print(f"y: {self.y.shape}")
        self.plot_correlation(df)
        self.plot_scatter(df)
        
    def plot_correlation(self, df):
        st.subheader('Correlation Matrix')
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
        st.pyplot(fig)
    
    def plot_scatter(self, df):
        st.subheader('Scatter Plot: radius_mean vs texture_mean')
        malignant_data = df[df['diagnosis'] == 1]  # Extract malignant data
        benign_data = df[df['diagnosis'] == 0] # Extract benign data
        fig2 = plt.figure(figsize=(10, 8))
        sns.scatterplot(x='radius_mean', y='texture_mean', data=malignant_data, label='Malignant', color='red')
        sns.scatterplot(x='radius_mean', y='texture_mean', data=benign_data, label='Benign', color='green')
        plt.title('radius_mean vs texture_mean')
        plt.legend()
        st.pyplot(fig2)

    def add_parameter_ui(self):
        if self.classifier_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 25.0)
            self.params['C'] = C
        elif self.classifier_name == 'KNN':
            K = st.sidebar.slider('K', 1, 25)
            self.params['K'] = K
        else:
            pass


    def get_classifier(self):
        if self.classifier_name == 'SVM':
            self.clf  = SVC(C=self.params['C'])
        elif self.classifier_name == 'KNN':
            self.clf  = KNeighborsClassifier(n_neighbors=self.params['K'])
        else:
            self.clf  = GaussianNB()

    def generate(self):
        self.get_classifier()
        #### CLASSIFICATION ####
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy =', acc)

        #### PLOT DATASET ####
        # Project the data onto the 2 primary principal components
        pca = PCA(2)
        X_projected = pca.fit_transform(self.X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig = plt.figure()
        plt.scatter(x1, x2,
                c=self.y, alpha=0.8,
                cmap='viridis')
        st.pyplot(fig)
