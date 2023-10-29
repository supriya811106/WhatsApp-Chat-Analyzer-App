import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px

# Suppressing the Streamlit warning
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_css(css_path):
    with open(css_path, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css("style.css")

# Main header and logo
st.image("images/logo.png", use_column_width=True)
st.title('WhatsApp Chat Analyzer')
st.write("""
Welcome to **Wanalyicia** - Your Personal WhatsApp Chat Insight Engine!
Dive deep into your chat narratives and discover patterns you never noticed before. From sentiment landscapes to user engagement metrics, 
Wanalyicia unveils a spectrum of data-driven revelations from your chats.
""")

st.write('**Guidelines to Begin:**')
st.write("""
1. Export your WhatsApp chat Without Media.
2. Upload the exported chat file.
3. Choose the user and the type of analysis.
4. View insights and charts based on your chat data.
""")

st.write("---")

st.sidebar.image("images/applogo.png", use_column_width=True)
uploaded_file = st.sidebar.file_uploader("Upload Exported Chat", type=["txt", "csv"])

if uploaded_file:
    # To read file as bytes
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique Users
    user_list = df['username'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall Users")

    # Adding a search box for selecting the user
    search_user = st.sidebar.text_input("Search for a User", "")

    # Filter the user list based on the search input
    filtered_users = [user for user in user_list if search_user.lower() in user.lower()]

    # Creating a select box with the filtered user list
    selected_user = st.sidebar.selectbox("Select The User", filtered_users)
    if selected_user == "Overall Users":
        analysis_menu = ["User Statistics", "Sentiment Analysis", "Advanced NLP Analysis", "Comparative Analysis", "User Activity", "Overall User Activity Analysis",
                         "Word and Emoji Analysis", "Timeline Analysis"]
    else:
        analysis_menu = ["User Statistics", "Sentiment Analysis", "Advanced NLP Analysis", "User Activity", "Word and Emoji Analysis",
                         "Timeline Analysis"]
    st.sidebar.header("Analysis Options")
    choice = st.sidebar.selectbox("Select Analysis Type", analysis_menu, index=0)

    # Sentiment Analysis
    if choice == "Sentiment Analysis":

        # Choose sentiment analysis method
        method = st.sidebar.selectbox("Choose sentiment analysis method", ["textblob", "vader"])

        if st.sidebar.button("Show Sentiment Analysis", key="sentiment_analysis_button"):  # Add a unique key
            if selected_user != 'Overall Users':
                df = df[df['username'] == selected_user]
            df['Sentiment'] = df['message'].apply(lambda x: helper.extract_sentiment(x, method))

            st.subheader("Sentiment Distribution")
            sentiment_distribution = df['Sentiment'].value_counts()
            fig = px.bar(sentiment_distribution, labels={'index': 'Sentiment', 'value': 'Count'})
            st.plotly_chart(fig)

            st.write("---")

            # Sentiment Trends Over Time
            if 'date' in df.columns:
                st.subheader("Sentiment Trends Over Time")
                sentiment_over_time = df.groupby(['date', 'Sentiment']).size().reset_index(name='Counts')
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.lineplot(data=sentiment_over_time, x='date', y='Counts', hue='Sentiment', ax=ax)
                st.pyplot(fig)

    elif choice == "Comparative Analysis":
        st.subheader("Comparative Analysis between Users")
        users_to_compare = st.multiselect("Select users for comparison", user_list)

        st.write("---")

        if users_to_compare:
            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            selected_range = st.slider("Select Time Range", min_date, max_date, (min_date, max_date))

            st.write("---")

            if st.sidebar.button("Show Comparative Analysis", key="comparative_analysis_button"):
                # Convert the selected_dates to numpy datetime64
                start_date = np.datetime64(selected_range[0])
                end_date = np.datetime64(
                    selected_range[1] + pd.Timedelta(days=1))  # To include the end date in the range

                date_filtered_df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date < end_date)]
                user_filtered_df = date_filtered_df[date_filtered_df["username"].isin(users_to_compare)]
                users_activity = user_filtered_df["username"].value_counts()
                st.bar_chart(users_activity)

    elif st.sidebar.button("Start Analysis"):
        # User Statistics
        if choice == "User Statistics":
            # Fetching stats
            total_messages, total_words, total_media_messages, total_url, total_emoji, deleted_message, edited_messages, shared_contact, shared_location = helper.fetch_stats(
                selected_user, df)
            st.markdown("### Total Messages Shared: ")
            st.write(f"<div class='big-font'>{total_messages}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Words Shared: ")
            st.write(f"<div class='big-font'>{total_words}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Media Shared: ")
            st.write(f"<div class='big-font'>{total_media_messages}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Link Shared: ")
            st.write(f"<div class='big-font'>{total_url}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Emoji Shared: ")
            st.write(f"<div class='big-font'>{total_emoji}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Deleted Message: ")
            st.write(f"<div class='big-font'>{deleted_message}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Edited Message: ")
            st.write(f"<div class='big-font'>{edited_messages}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Contact Shared: ")
            st.write(f"<div class='big-font'>{shared_contact}</div>", unsafe_allow_html=True)

            st.write("---")

            st.markdown("### Total Location Shared: ")
            st.write(f"<div class='big-font'>{shared_location}</div>", unsafe_allow_html=True)

        # Advanced NLP
        elif choice == "Advanced NLP Analysis":
            st.subheader("TF-IDF Analysis")

            # Vectorizing the messages using TF-IDF
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tfidf = vectorizer.fit_transform(df['message'])
            words = vectorizer.get_feature_names_out()

            # Displaying top 5 words with the highest TF-IDF score for demonstration
            top_n = 5
            row_id = np.argmax(tfidf.toarray(), axis=0)
            top_words = [(words[i], tfidf[row_id[i], i]) for i in np.argsort(-tfidf.toarray().sum(axis=0))[:top_n]]
            st.write("Top 5 words based on TF-IDF scores:", top_words)

            st.write("---")

            st.subheader("LDA Topic Modeling")

            # Creating a CountVectorizer (BoW) representation
            vectorizer_bow = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            bow = vectorizer_bow.fit_transform(df['message'])
            words_bow = vectorizer_bow.get_feature_names_out()

            num_topics = 5

            lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
            lda.fit(bow)

            # Displaying top 5 words associated with each topic
            for i, topic in enumerate(lda.components_):
                top_words_array = topic.argsort()[-top_n:][::-1]
                topic_words = [words_bow[i] for i in top_words_array]
                st.write(f"Topic {i + 1}: {' | '.join(topic_words)}")

        # User Activity
        elif choice == "User Activity":
            if selected_user == 'Overall Users':
                top, bottom = helper.most_least_busy_users(df)

                st.subheader('Most Active Users')
                st.bar_chart(top)

                st.write("---")

                st.subheader('Least Active Users')
                st.bar_chart(bottom)

                st.write("---")

            # Creating a line chart to visualize user activity over time
            st.subheader("User Activity Over Time")
            user_activity = helper.user_activity_over_time(selected_user, df)
            st.line_chart(user_activity)

            st.write("---")

            # Week Activity Map
            st.subheader("Week Activity Map")
            week_activity_data = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 6))
            week_activity_data.sort_index().plot(kind='bar', ax=ax)
            ax.set_title("Activity Throughout the Week")
            ax.set_ylabel("Number of Messages")
            ax.set_xlabel("Day of the Week")
            st.pyplot(fig)

            st.write("---")

            # Month Activity Map
            st.subheader("Month Activity Map")
            month_activity_data = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 6))
            month_activity_data.sort_index().plot(kind='bar', ax=ax)
            ax.set_title("Activity Throughout the Month")
            ax.set_ylabel("Number of Messages")
            ax.set_xlabel("Month")
            st.pyplot(fig)

            st.write("---")

            # Activity Heatmap
            st.subheader("Activity Heatmap")
            heatmap_data = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".0f", ax=ax)
            ax.set_title("Activity Heatmap: Day vs. Period")
            st.pyplot(fig)

        elif choice == "Overall User Activity Analysis":
            user_activity_df = helper.user_activity_in_chat(df)
            st.header("Activity Analysis of Each User in a Group Chat:")

            st.subheader("Total Messages Sent By The User:")
            st.bar_chart(user_activity_df, x='username', y='Total_Messages')

            st.write("---")

            st.subheader("Total Words Sent By The User:")
            st.bar_chart(user_activity_df, x='username', y='Total_Words')

            st.write("---")

            st.subheader("Percentage of Messages Sent By The User:")
            st.bar_chart(user_activity_df, x='username', y='Percentage')

            st.write("---")

            st.subheader("Total Media Shared By The User:")
            st.bar_chart(user_activity_df, x='username', y='Media_Shared')

            st.write("---")

            st.subheader("Total Links Shared By The User:")
            st.bar_chart(user_activity_df, x='username', y='Links_Shared')

            st.write("---")

            st.subheader("Total Emojis Shared By The User:")
            st.bar_chart(user_activity_df, x='username', y='Emojis_Shared')

            st.write("---")

            st.subheader("Total Deleted Messages By Each User:")
            st.bar_chart(user_activity_df, x='username', y='Deleted_Messages')

            st.write("---")

            st.subheader("Total Edited Messages By Each User:")
            st.bar_chart(user_activity_df, x='username', y='Edited_Messages')

            st.write("---")

            st.subheader("Total Contacts Shared By The User:")
            st.bar_chart(user_activity_df, x='username', y='Shared_Contacts')

            st.write("---")

            st.subheader("Total Locations Shared By The User:")
            st.bar_chart(user_activity_df, x='username', y='Shared_Locations')

        # Word and Emoji Analysis
        elif choice == "Word and Emoji Analysis":
            wordcloud_image = helper.create_wordcloud(selected_user, df)
            # Convert WordCloud to Image
            wc_img = Image.new("RGB", (wordcloud_image.width, wordcloud_image.height))
            wc_array = np.array(wordcloud_image)
            wc_img.paste(Image.fromarray(wc_array), (0, 0))
            st.subheader("Word Cloud")
            st.image(wc_img, use_column_width=True, caption="Word Cloud of Chat")

            st.write("---")

            # Emoji Analysis
            emoji_df = helper.emoji_helper(selected_user, df)
            st.subheader("Emoji Analysis:")
            st.dataframe(emoji_df)

            st.write("---")

            # Plot top 5 emojis
            fig, ax = plt.subplots()
            ax.pie(emoji_df['Frequency'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f")
            st.pyplot(fig)

        # Timeline Analysis
        elif choice == "Timeline Analysis":
            monthly_timeline = helper.monthly_timeline(selected_user, df)
            daily_timeline = helper.daily_timeline(selected_user, df)

            st.subheader("Monthly Timeline")
            st.line_chart(monthly_timeline)

            st.write("---")

            st.subheader("Daily Timeline")
            st.line_chart(daily_timeline)

st.sidebar.header("We Value Your Feedback")

# Dropdown for selecting feedback
was_helpful = st.sidebar.selectbox("Did you find our insights useful?", ["Please choose an option", "Yes, very helpful", "Somewhat helpful", "Not helpful"])

# Text area for additional feedback
additional_feedback = st.sidebar.text_area("Any suggestions or feedback?")

# Submit feedback
if st.sidebar.button("Submit Feedback"):
    # Process feedback (save to a database or send to an email)
    st.sidebar.success("Thank you for your feedback!")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2023 Wanalyicia. All rights reserved.")

