import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

#Loading the text data
    
df = pd.read_csv('./documents.csv', names=["message"])
print(df.head())
print("\n\n")

#Vocabulary to calculate TF-IDF for
voc=['free','click', 'here','visit','open', 'attachment','call', 'this' ,'number',
'money','Out','extra','offer','available','pension','opportunity','chance','investment']

#applying porter stemmer to the vocabulary
porter = PorterStemmer()
voc_stemmed = [porter.stem(word) for word in voc]

#function to preprocess the text
def preprocess(text):
    #text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens=word_tokenize(tokens)
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
        
    return stemmed 

#Applying CountVectorizer to prepare VSM model (document-term matrix)
bow_transformer = CountVectorizer(analyzer=preprocess,vocabulary=voc_stemmed).fit(df['message'])
print(bow_transformer.vocabulary_)
print("\n\n")

#Genrating the matrix and printing
messages_bow = bow_transformer.transform(df['message'])
print('sparse matrix shape:', messages_bow.toarray())
print("\n")
print('sparse matrix shape:', messages_bow.shape)
print("\n\n")
print("TF-IDF")

#Applying TF-IDF transformer to genearte TF-IDF of the terms in documents
tfidf_transformer = TfidfTransformer().fit(messages_bow)

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf)

