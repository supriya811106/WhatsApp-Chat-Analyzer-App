from wordcloud import WordCloud
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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


def most_least_busy_users(df):
    message_counts = df['username'].value_counts()
    # Getting top 5 users (most active)
    top_users = message_counts.head(5)
    # Getting bottom 5 users (least active)
    bottom_users = message_counts.tail(5)
    return top_users, bottom_users


def user_activity_in_chat(df):
    # Adjusted the media counting logic
    df['Media_Shared'] = df['message'].apply(lambda x: 1 if 'Media' in x else 0)

    # Count shared Contacts
    phone_pattern = r'\+?\d{2,4}[\s-]?\d{10}'

    # Count shared Location
    location_pattern = r'//maps\.google\.com/\?q=\d+\.\d+,\d+\.\d+'

    # Basic aggregations
    agg_df = df.groupby('username').agg(
        Total_Messages=pd.NamedAgg(column='message', aggfunc='size'),
        Total_Words=pd.NamedAgg(column='total_word', aggfunc='sum'),
        Media_Shared=pd.NamedAgg(column='Media_Shared', aggfunc='sum'),
        Links_Shared=pd.NamedAgg(column='url_count', aggfunc='sum'),
        Emojis_Shared=pd.NamedAgg(column='emoji_count', aggfunc='sum'),
        Deleted_Messages=pd.NamedAgg(column='message',
                                     aggfunc=lambda x: x.str.contains("This message was deleted").sum()),
        Edited_Messages=pd.NamedAgg(column='message',
                                    aggfunc=lambda x: x.str.contains("<This message was edited>").sum()),
        Shared_Contacts=pd.NamedAgg(column='message',
                                    aggfunc=lambda x: x.str.contains(phone_pattern, case=False).sum() + x.str.contains(
                                        '.vcf', case=False).sum()),
        Shared_Locations=pd.NamedAgg(column='message',
                                     aggfunc=lambda x: x.str.contains(location_pattern, case=False).sum())
    ).reset_index()

    # Percentage calculation
    agg_df['Percentage'] = round((agg_df['Total_Messages'] / df.shape[0]) * 100, 2)

    # Desired column order
    column_order = [
        'username',
        'Total_Messages',
        'Percentage',
        'Total_Words',
        'Media_Shared',
        'Links_Shared',
        'Emojis_Shared',
        'Deleted_Messages',
        'Edited_Messages',
        'Shared_Contacts',
        'Shared_Locations'
    ]
    # Sort dataframe by Total_Messages in descending order
    agg_df = agg_df.sort_values(by='Total_Messages', ascending=False)

    # Reset index and increment by 1
    agg_df = agg_df.reset_index(drop=True)
    agg_df.index = agg_df.index + 1

    return agg_df


def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    temp = df[df['message'] != '<Media omitted>']
    temp = temp[temp['message'] != "This message was deleted"]
    temp = temp[temp['message'] != '<This message was edited>']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall Users':
        df = df[df['username'] == selected_user]

    temp = df[df['message'] != '<Media omitted>']
    temp = temp[temp['message'] != "This message was deleted"]
    temp = temp[temp['message'] != '<This message was edited>']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


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


def extract_sentiment(text, method):
    if method == "textblob":
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
    elif method == "vader":
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)
        if score['compound'] > 0.05:
            return 'positive'
        elif score['compound'] < -0.05:
            return 'negative'
        else:
            return 'neutral'
