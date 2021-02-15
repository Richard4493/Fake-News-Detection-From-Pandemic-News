import streamlit as st
import matplotlib.pyplot as plt

from FakeNewsDetection import fake_news_detection
if __name__ == '__main__':

    fk = fake_news_detection("file1.csv")
    data = fk.compare()
    a=data['accuracy']
    st.markdown("""
    <style>
    body {
        color: #4e8d7c;
        background-color: #e6e6e6;
    }
    </style>
        """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    sidebar .sidebar-content {
        background-color: #ffff !important;
    }
    </style>
        """, unsafe_allow_html=True)
    html_temp = """
                	<div style="background-color:black;padding:10px">
               	<h1 style="color:white;text-align:left;">FAKE NEWS DETECTION </h1>
               	</div>
                	"""
    st.markdown(html_temp, unsafe_allow_html=True)


    st.sidebar.write("Dataset used : covid_news Dataset")
    claasifier_name = st.sidebar.selectbox("Select The classifier", ("logistic regression","PAC","lin svm", "poly svm","RBF svm", "random forest", "decision tree"))


    def get_classifier(clf_name):
        if clf_name == "logistic regression":
            clf = a[0]
        elif clf_name == "PAC":
            clf = a[1]
        elif clf_name == "lin svm":
            clf = a[2]
        elif clf_name == "poly svm":
            clf = a[3]
        elif clf_name == "RBF svm":
            clf = a[4]
        elif clf_name == "random forest":
            clf = a[5]
        elif clf_name=="decision tree":
            clf=a[6]

        return clf


    opt = st.sidebar.checkbox("CLick to view the accuracy of the classifier")
    if opt == True :
        clf=get_classifier(claasifier_name)
        st.write(f"classifier={claasifier_name}")
        st.write(f"accuracy = {clf}")
    option = st.sidebar.checkbox("Do you want to see the accuracy graph")
    # st.write("You have selected:",option)
    if option == True:
        lr_per =a[0]
        pac_per=a[1]
        lin_per=a[2]
        poly_per=a[3]
        rbf_per=a[4]
        rf_per=a[5]
        dt_per=a[6]
        x = ["LR","pac", "lin svm", "poly svm","RBF svm","RF", "DT"]
        h = [lr_per,pac_per,lin_per, poly_per,rbf_per, rf_per, dt_per]
        plt.figure()
        plt.bar(x, h)
        plt.xlabel("Classifier")
        plt.ylabel("Accuracy")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

