import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Analyze Your WhatsApp Chat")
uploaded_file = st.sidebar.file_uploader("Upload Exported Chat")
if uploaded_file is not None:
    # To read file as bytes
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Fetch unique Users
    user_list = df['username'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall Users")

    # Add a search box for selecting the user
    search_user = st.sidebar.text_input("Search for a User", "")

    # Filter the user list based on the search input
    filtered_users = [user for user in user_list if search_user.lower() in user.lower()]

    # Create a select box with the filtered user list
    selected_user = st.sidebar.selectbox("Select The User", filtered_users)

    if st.sidebar.button("Show Analysis"):
        st.title("Analysis of Your Whatsapp Chat 📈")
        total_messages, total_words, total_media_messages, total_url, total_emoji, deleted_message, edited_messages, shared_contact, shared_location = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics:")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Total Messages")
            st.title(total_messages)

        with col2:
            st.header("Total Words")
            st.title(total_words)

        with col3:
            st.header("Media Shared")
            st.title(total_media_messages)

        # Draw a line separator before col4
        st.markdown("---")

        col4, col5, col6 = st.columns(3)

        with col4:
            st.header("Links Shared")
            st.title(total_url)

        with col5:
            st.header("Emoji Shared")
            st.title(total_emoji)

        with col6:
            st.header("Deleted Message")
            st.title(deleted_message)

        # Draw a line separator before col4
        st.markdown("---")

        col7, col8, col9 = st.columns(3)

        with col7:
            st.header("Edited Messages")
            st.title(edited_messages)

        with col8:
            st.header("Shared Contacts")
            st.title(shared_contact)

        with col9:
            st.header("Shared Locations")
            st.title(shared_location)

        # Draw a line separator before col4
        st.markdown("---")

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall Users':
            top, bottom = helper.most_least_busy_users(df)
            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots()
                st.title('Most Active Users:')
                ax1.bar(top.index, top.values, color='#4B527E')
                plt.xticks(rotation='vertical')
                st.pyplot(fig1)
            with col2:
                st.title('Least Active Users:')
                fig2, ax2 = plt.subplots()
                ax2.bar(bottom.index, bottom.values, color='#2D4356')
                plt.xticks(rotation='vertical')
                st.pyplot(fig2)

            st.markdown("---")

            new_df = helper.user_activity_in_chat(df)
            st.title("User Activity Summary:")
            new_df_sorted = new_df.sort_values(by='Total_Messages', ascending=False)
            st.dataframe(new_df_sorted)

        st.markdown("---")
        # WordCloud
        st.title("Word Cloud of Chat:")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        st.markdown("---")

        # most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most Common Words:')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis:")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['Frequency'].head(), labels=emoji_df['Emoji'].head(), autopct="%0.2f")
            st.pyplot(fig)

        # monthly timeline
        st.title("Monthly Timeline:")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline:")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['date'], daily_timeline['message'], color='black')  # Updated here
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map:')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day:")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month:")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map:")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)