import json
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyserini.search import get_topics, SimpleSearcher
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import hashlib

def preprocess_text(text):
    # Define the stopwords to remove along with punctuation
    stop_words = set(stopwords.words('english')).union(set(string.punctuation))
    
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Tokenize the text, remove stopwords, and convert to lowercase
    tokens = [word.lower() for word in word_tokenize(text) if word.lower() not in stop_words]

    # Apply stemming and lemmatization
    stemmed = [stemmer.stem(word) for word in tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]

    # Return the lemmatized tokens
    return lemmatized


# Define the function to get the embeddings
def compute_embedding(sentence, word2vec_model, max_tokens=None):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Optionally limit the number of tokens, adding padding if necessary
    tokens = tokens if not max_tokens else tokens[:max_tokens] + ['[PAD]'] * max(0, max_tokens - len(tokens))
    # Get the word embeddings for each token
    word_embeddings = [word2vec_model.wv[token.lower()] for token in tokens if token.lower() in word2vec_model.wv]

    # If no valid word embeddings found, return zero vector
    if not word_embeddings:
        return np.zeros(word2vec_model.vector_size * max_tokens)
    
    # If max_tokens is specified
    if max_tokens:
        # Pad the embeddings with zero vectors
        padding = [np.zeros(word2vec_model.vector_size) for _ in range(max_tokens - len(word_embeddings))]
        # Extend the embeddings with the padding
        word_embeddings.extend(padding)
        # Return the embeddings
        return np.concatenate(word_embeddings, axis=0)
    # Otherwise
    else:
        # Return the average of the embeddings
        return np.mean(word_embeddings, axis=0)


# Define the function that builds the dictionaries of queries and documents
def build_dicts(max_topics=2, max_docs=3):
    # Get the topics
    topics = get_topics('msmarco-passage-dev-subset')
    # Initialize the searcher
    searcher = SimpleSearcher.from_prebuilt_index('msmarco-passage')

    # Initialize the dictionaries and the corpus
    queries = dict()
    documents = dict()
    corpus = list()

    # Optionally limit the number of topics processed
    topics_items = list(topics.items())[:max_topics] if max_topics else topics.items()

    # For each topic
    for query_id, topic_info in tqdm(topics_items, "Building dictionaries"):
        # Get the query embedding and update the queries dictionary
        queries[query_id] = {
            'raw': topic_info['title'],     # Raw query
            'docids_list': list()}          # List of correlated documents
        # Append the query embedding to the w2v data
        corpus.append(preprocess_text(queries[query_id]['raw']))

        # Perform the search
        hits = searcher.search(queries[query_id]['raw'], max_docs)

        # For each document retrieved
        for hit in hits:
            # If the document is not already in the documents dictionary
            if hit.docid not in documents:
                # Retrieve document content as a string and update the documents dictionary
                documents[hit.docid] = {'raw': json.loads(searcher.doc(hit.docid).raw())['contents']}   
                # Append the document embedding to the w2v data
                corpus.append(preprocess_text(documents[hit.docid]['raw']))        

            # Append the document id to the list of correlated documents
            queries[query_id]['docids_list'].append(hit.docid)

    # Return the three dictionaries
    return queries, documents, corpus


# Define the function that generates the docids
def generate_semantic_docid(document, max_length=12):
    # Extract key features (e.g., top keywords)
    vectorizer = TfidfVectorizer(max_features=5)  # Adjust as needed
    tfidf_matrix = vectorizer.fit_transform([document])
    feature_names = vectorizer.get_feature_names_out()
    sorted_features = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    key_features = [feature_names[index] for index in sorted_features[:3]]  # Top 3 features

    # Encode features as numbers
    encoder = LabelEncoder()
    encoded_features = encoder.fit_transform(key_features)

    # Combine encoded features into a single number
    combined_features_number = int(''.join(map(str, encoded_features)))

    # Create a unique part based on the document's content
    # Use SHA-256 instead of MD5 and take a longer part of the hash
    unique_part = int(hashlib.sha256(document.encode()).hexdigest()[:max_length], 16)  

    combined_number = int(str(combined_features_number) + str(unique_part))
    docid = str(combined_number)[:max_length].zfill(max_length)

    return docid


# Define the function that generates the semantic docids
def generate_semantic_docids(documents, max_length=12):
    # Initialize mapping
    semantic_to_original = dict()

    # Iterate over documents
    for docid, content in tqdm(documents.items(), desc='Creating semantic docids'):
        # Tokenizing and encoding the document text (we add the docid to enforce uniqueness)
        preprocessed_text = " ".join(preprocess_text(content['raw'] + ' ' + docid))
        # Generate semantic docid
        semantic_docid = generate_semantic_docid(preprocessed_text, max_length)
        # Make sure the semantic docid is unique
        assert semantic_docid not in semantic_to_original
        # Store mapping
        semantic_to_original[semantic_docid] = docid

    return semantic_to_original