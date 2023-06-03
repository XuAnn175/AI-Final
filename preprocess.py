from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer
import string

def remove_stopwords(text: str) -> str:
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

def preprocessing_function(text: str) -> str:
    text = text.lower() # lower cases
    text = remove_stopwords(text) # Remove stopwords
    text = "".join([char for char in text if (char not in string.punctuation)]) # Remove punctuations
    english_stemmer = SnowballStemmer(language='english') # SnowballStemmer
    text = " ".join([english_stemmer.stem(i) for i in text.split()]) 
    preprocessed_text = text

    return preprocessed_text

if __name__ == "__main__":
    tst = "I'm sorry. I've joined the league of peoples that dont keep in touch. You means a great deal to me. You has been a friend at all times even at great personal cost. Do have a great week."
    print(preprocessing_function(tst))