import streamlit as st
from FakeNewsDetection import fake_news_detection
fk = fake_news_detection("corona_fake_news.csv")
def app() :
        html_temp = """
            	<div style="background-color:black;padding:10px">
            	<h1 style="color:white;text-align:left;">FAKE NEWS DETECTION </h1>
            	</div>
            	"""
        st.markdown(html_temp, unsafe_allow_html=True)
        # st.subheader("ML App with Streamlit")
        html_temp = """
    	<div style="background-color:white;padding:5px">
    	<h2 style="color:black;text-align:center;">Streamlit ML App </h2>
    	</div>
    	"""
        st.markdown(html_temp, unsafe_allow_html=True)


        activity = ['Prediction']
        choice = st.sidebar.selectbox("Select Activity", activity)

        if choice == 'Prediction':

            news_title = st.text_area("Enter News Title", "Type Here")
            news_text = st.text_area("Enter News Text", "Type Here")

        if st.button("Predict"):
            result=fk.predict(news_title,news_text)
            st.success("News Categorized as:: {}".format(result))
if __name__ == '__main__':
	app()