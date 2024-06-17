import json
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_data(data):
    """Extract text and category from the data."""
    return [{'text': entry['text'], 'category': entry['category']} for entry in data]

def get_insights(df):
    """Preprocess the DataFrame by adding new features."""
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(x.split())))
    df['uppercase_word_count'] = df['text'].apply(lambda x: sum(1 for word in x.split() if word.isupper()))
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    df['contains_link'] = df['text'].apply(lambda x: 'http' in x)
    df['contains_parentheses'] = df['text'].apply(lambda x: '(' in x or ')' in x)
    df['contains_quotation_marks'] = df['text'].apply(lambda x: '"' in x or "'" in x)
    df['contains_5G'] = df['text'].apply(lambda x: '5G' in x or '5 G' in x)
    df['contains_bill_gates'] = df['text'].apply(lambda x: 'Bill Gates' in x)
    return df

def plot_histogram(df, category, title, xlabel, ylabel):
    """Plot a histogram of text length by category."""
    plt.figure(figsize=(10, 5))
    df[df['category'] == 'CONSPIRACY'][category].hist(alpha=0.5, label='CONSPIRACY', bins=30)
    df[df['category'] == 'CRITICAL'][category].hist(alpha=0.5, label='CRITICAL', bins=30)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_bar_chart(df, feature, title, ylabel):
    """Plot a bar chart for a feature by category."""
    feature_by_category = df.groupby('category')[feature].mean()
    feature_by_category.plot(kind='bar', color=['blue', 'green'])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()