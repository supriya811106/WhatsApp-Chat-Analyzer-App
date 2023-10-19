import re
import pandas as pd
import emoji
from urlextract import URLExtract


def preprocess(data):
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[apAP][mM]\s-\s)'
    parts = re.split(pattern, data)

    # Filter out empty strings and timestamp-related strings
    messages = [part for part in parts if part and not re.match(pattern, part)]
    dates = re.findall(pattern, data)
    dates = [date.replace('\u202f', ' ') for date in dates]

    df = pd.DataFrame({'user_message': messages, 'date': dates})
    # convert message_date type
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y, %I:%M %p - ')

    # Format 'date' column as desired
    df['new_date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M %p')

    patternforusername = r'([^:]+):'

    # Use str.extract to create a new 'username' column
    df['username'] = df['user_message'].str.extract(patternforusername)

    # Remove the username from 'user_message' and create a new 'message' column
    df['message'] = df['user_message'].str.replace(patternforusername, '', regex=True).str.strip()

    df.drop(columns=['user_message'], inplace=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    df['new_date'] = pd.to_datetime(df['new_date'], format='%Y-%m-%d %I:%M %p')

    # Create separate date and time columns
    df['date'] = df['new_date'].dt.date

    # Format the time as '7:24 PM' and create a new 'time' column
    df['time'] = df['new_date'].dt.strftime('%I:%M %p')

    df.drop(columns=['new_date'], inplace=True)
    df = df.dropna(subset=['username'])

    def count_words(text):
        words = text.split()
        return len(words)

    # Apply the function to the 'message' column to count words
    df['total_word'] = df['message'].apply(count_words)

    extractor = URLExtract()
    # Function to count URLs in a message

    def count_urls(text):
        urls = extractor.find_urls(text)
        return len(urls)

    df['url_count'] = df['message'].apply(count_urls)

    # Function to count emojis in a message
    def count_emojis(message):
        return emoji.emoji_count(message)

    # Apply the function to count emojis in each message
    df['emoji_count'] = df['message'].apply(count_emojis)

    def assign_period(hour):
        if 0 <= hour < 6:
            return 'Night'
        elif 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'

    df['period'] = df['hour'].apply(assign_period)

    return df
