import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import inflect


# Define pre-processing function
def preprocess_with_stopwords_with_lemmatization(text):
    # Lowercase
    text = text.lower()
    # Replace "5G" or "5 G" with "5 G"
    text = re.sub(r'\b(5\s?)g\b', '5 g', text)
    # Strip white space
    text = text.strip()
    # Remove spaces between specific punctuation marks
    text = re.sub(r'\s*([.,/!])\s*', r'\1', text)
    # Remove URLs
    text = re.sub(r'https?\s*:\s*\/\s*\/\s*\S+', '', text)
    # Initialize inflect engine
    p = inflect.engine()
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation except exclamation mark
    text = re.sub(r'[^\w\s!]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    text = ' '.join(tokens)
    return text


# Define pre-processing function
def preprocess_without_stopwords_with_lemmatization(text):
    # Lowercase
    text = text.lower()    
    # Replace "5G" or "5 G" with "5 G"
    text = re.sub(r'\b(5\s?)g\b', '5 g', text)
    # Strip white space
    text = text.strip()
    # Remove spaces between specific punctuation marks
    text = re.sub(r'\s*([.,/!])\s*', r'\1', text)
    # Remove URLs
    text = re.sub(r'https?\s*:\s*\/\s*\/\s*\S+', '', text)
    # Initialize inflect engine
    p = inflect.engine()
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation except exclamation mark
    text = re.sub(r'[^\w\s!]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    text = ' '.join(tokens)

    return text

# Define pre-processing function
def preprocess_with_stopwords_without_lemmatization(text):
    # Lowercase
    text = text.lower()    
    # Replace "5G" or "5 G" with "5 G"
    text = re.sub(r'\b(5\s?)g\b', '5 g', text)
    # Strip white space
    text = text.strip() 
    # Remove spaces between specific punctuation marks
    text = re.sub(r'\s*([.,/!])\s*', r'\1', text)
    # Remove URLs
    text = re.sub(r'https?\s*:\s*\/\s*\/\s*\S+', '', text)
    # Initialize inflect engine
    p = inflect.engine()
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation except exclamation mark
    text = re.sub(r'[^\w\s!]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Join tokens back to string
    text = ' '.join(tokens)
    return text

# Define pre-processing function
def preprocess_without_stopwords_without_lemmatization(text):
    # Lowercase
    text = text.lower()    
    # Replace "5G" or "5 G" with "5 G"
    text = re.sub(r'\b(5\s?)g\b', '5 g', text)
    # Strip white space
    text = text.strip()
    # Remove spaces between specific punctuation marks
    text = re.sub(r'\s*([.,/!])\s*', r'\1', text)
    # Remove URLs
    text = re.sub(r'https?\s*:\s*\/\s*\/\s*\S+', '', text)
    # Initialize inflect engine
    p = inflect.engine()
    # Replace numbers with words
    text = re.sub(r'\d+', lambda x: p.number_to_words(x.group()), text)
    # Remove punctuation except exclamation mark
    text = re.sub(r'[^\w\s!]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back to string
    text = ' '.join(tokens)
    return text