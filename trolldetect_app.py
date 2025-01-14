import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

def data_page():
   # App title
    st.title("Data of Youtube Comments During the Malaysian Election")
    # Load dataset
    try:
        data = pd.read_csv("yt_data.csv")  # Replace with your dataset's filename
        st.write("### Dataset Preview")
        st.dataframe(data.head())  # Show first few rows

        # Basic statistics
        st.write("### Dataset Statistics")
        st.write(f"**Total Comments:** {len(data)}")
        st.write(f"**Troll Comments:** {sum(data['Status'] == 1)}")  # Assuming 'label' column
        st.write(f"**Non-Troll Comments:** {sum(data['Status'] == 0)}")
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload the dataset.")

    # Troll vs Non-Troll Distribution
    st.write("### Troll vs Non-Troll Distribution")
    status_counts = data['Status'].value_counts()
    st.bar_chart(status_counts)


    st.write("### Word Cloud for Specific Comment Type")
    wc_filter = st.selectbox("Generate Word Cloud for", ["Overall Comments", "Non-Troll Comments", "Troll Comments"])

    if wc_filter == "Non-Troll Comments":
        text = " ".join(data[data['Status'] == 0]['Comment'].dropna())
    elif wc_filter == "Troll Comments":
        text = " ".join(data[data['Status'] == 1]['Comment'].dropna())
    else:
        text = " ".join(data['Comment'].dropna())  # All comments

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


def detection_page():

    # Load the Logistic Regression model and TF-IDF vectorizer using Joblib
    svm_model = joblib.load("svm_troll_detection_model.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")

    # App title
    st.title("Troll Detection Model For the Malaysian Election")

    # Input from user
    user_input = st.text_area("Enter a comment to analyze:")

    if st.button("Analyze"):
        if user_input.strip():
            # Preprocess input
            processed_input = user_input.lower()  # Adjust as per your preprocessing function

            # Transform input using TF-IDF
            input_vector = tfidf.transform([processed_input])

            # Predict with SVM model
            prediction = svm_model.predict(input_vector)[0]
            probabilities = svm_model.predict_proba(input_vector)[0]

            # Display result
            if prediction == 1:
                st.write("🚨 This comment is likely a troll.")
                st.write(f"Confidence: {probabilities[1]:.2%}")
            else:
                st.write("✅ This comment is not a troll.")
                st.write(f"Confidence: {probabilities[0]:.2%}")
        else:
            st.write("Please enter a comment.")

# Main Streamlit app
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Explore Data', 'Troll Detection'])

    if page == 'Explore Data':
        data_page()
    elif page == 'Troll Detection':
        detection_page()

if __name__ == '__main__':
    main()

