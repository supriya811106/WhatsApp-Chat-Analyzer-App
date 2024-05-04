from wordcloud import WordCloud
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import emoji
from collections import Counter


def fetch_stats(selected_user, df):
    # For a specific User
    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    # Count total no of messages
    total_messages = df.shape[0]

    # Count total no of words
    total_word_count = df['total_word'].sum()

    # Count total no of media messages
    total_media_messages = df[df['message'] == '<Media omitted>'].shape[0]

    # Total count of URLs in the entire DataFrame
    total_url_count = df['url_count'].sum()

    # Total count of Emojis in the entire DataFrame
    total_emoji_count = df['emoji_count'].sum()

    # Count total no of deleted message
    deleted_message = df[df['message'].str.contains("This message was deleted")]['message']

    # Count edited messages
    edited_messages = df[df['message'].str.contains("<This message was edited>")]['message']

    # Count shared Contacts
    phone_pattern = r'\+?\d{2,4}[\s-]?\d{10}'

    # Check for messages containing phone numbers or contact VCF files
    shared_contacts = df[df['message'].str.contains(phone_pattern, case=False) | df['message'].str.contains('.vcf', case=False)]['message']

    # Count shared Location
    location_pattern = r'//maps\.google\.com/\?q=\d+\.\d+,\d+\.\d+'
    shared_locations = df[df['message'].str.contains(location_pattern, case=False)]['message']

    return total_messages, total_word_count, total_media_messages, total_url_count, total_emoji_count, len(deleted_message), len(edited_messages), len(shared_contacts), len(shared_locations)

def extract_sentiment(message):
    analysis = TextBlob(message)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def perform_tfidf_analysis(messages):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(messages)
    words = vectorizer.get_feature_names_out()
    top_n = 5
    row_id = np.argmax(tfidf.toarray(), axis=0)
    top_words = [(words[i], tfidf[row_id[i], i]) for i in np.argsort(-tfidf.toarray().sum(axis=0))[:top_n]]
    return top_words


def perform_lda_analysis(messages, num_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    bow = vectorizer.fit_transform(messages)
    words = vectorizer.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(bow)
    topic_words = []
    for i, topic in enumerate(lda.components_):
        top_words_array = topic.argsort()[-5:][::-1]
        topic_list = [words[j] for j in top_words_array]
        topic_words.append(f"Topic {i + 1}: {' | '.join(topic_list)}")
    return topic_words


def perform_comparative_analysis(df, users_to_compare, start_date, end_date):
    # Ensure dates are handled as datetime64
    start_date = np.datetime64(start_date)
    end_date = np.datetime64(end_date + pd.Timedelta(days=1))  # Including the end date

    # Filter the DataFrame by date and selected users
    date_filtered_df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date < end_date)]
    user_filtered_df = date_filtered_df[date_filtered_df["username"].isin(users_to_compare)]

    # Count occurrences of each user
    users_activity = user_filtered_df["username"].value_counts()
    return users_activity


def most_least_busy_users(df):
    message_counts = df['username'].value_counts()
    # Getting top 5 users (most active)
    top_users = message_counts.head(5)
    # Getting bottom 5 users (least active)
    bottom_users = message_counts.tail(5)
    return top_users, bottom_users


def user_activity_over_time(selected_user, df):
    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    user_activity = df.groupby(['date', 'username'])['message'].count().unstack().fillna(0)

    return user_activity


def week_activity_map(selected_user, df):
    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = df['day'].value_counts().reindex(ordered_days).fillna(0)

    return day_counts


def month_activity_map(selected_user, df):
    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    user_heatmap = df.pivot_table(index='day', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def create_wordcloud(selected_user, df, stopwords_path='stop_hinglish.txt'):
    with open(stopwords_path, 'r') as f:
        stop_words = set(f.read().split())

    # Filter DataFrame according to user and message relevance
    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    # Exclude specific types of messages
    df = df[~df['message'].isin(['<Media omitted>', 'This message was deleted', '<This message was edited>'])]

    # Using WordCloud with stopwords
    wc = WordCloud(width=800, height=400, min_font_size=10, background_color='white', stopwords=stop_words)
    df_wc = wc.generate(' '.join(df['message']))  # Generate from all messages concatenated

    return df_wc.to_image()


def emoji_helper(selected_user, df):
    # Filter messages based on selected user (unless it's "Overall Users")
    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    # Generate the full set of emojis using emojize
    all_possible_emojis = set(emoji.emojize(alias) for alias in emoji.unicode_codes.EMOJI_DATA.keys())

    # Extract all emojis from the messages
    all_emojis = []
    for message in df['message']:
        message_str = str(message)
        all_emojis.extend([char for char in message_str if char in all_possible_emojis])

    # Create a dataframe with the frequency count of these emojis
    emoji_df = pd.DataFrame(Counter(all_emojis).most_common(), columns=['Emoji', 'Frequency'])

    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()  # Removed 'month_num'

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    timeline = df.groupby('date').count()['message'].reset_index()
    timeline = timeline.set_index('date')
    return timeline
