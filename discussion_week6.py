import re
import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from nltk.metrics import jaccard_distance
from scipy.cluster.hierarchy import dendrogram, linkage

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc


def main():
    # create corpus
    with open('week6_data/harry_potter.txt') as file1:
        harry_potter = file1.read()
    with open('week6_data/lotr.txt') as file2:
        lotr = file2.read()
    with open('week6_data/asoiaf.txt') as file3:
        asoiaf = file3.read()
    corpus = [harry_potter, lotr, asoiaf]
    # normalize doc
    normalize_corpus = np.vectorize(normalize_document)
    norm_corpus = normalize_corpus(corpus)
    # tf-idf
    tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True, smooth_idf=True)
    tv_matrix = tv.fit_transform(norm_corpus)
    tv_matrix = tv_matrix.toarray()
    # document similarity
    cosine_similarity_matrix = cosine_similarity(tv_matrix)
    cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, columns=['harry_potter', 'lotr', 'asoiaf'], index=['harry_potter', 'lotr', 'asoiaf'])
    print("Cosine Similarity\n", cosine_similarity_df)
    jaccard_similarity_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            jaccard_similarity_matrix[i][j] = jaccard_distance(set(tv_matrix[i]), set(tv_matrix[j]))
    jaccard_similarity_df = pd.DataFrame(jaccard_similarity_matrix, columns=['harry_potter', 'lotr', 'asoiaf'], index=['harry_potter', 'lotr', 'asoiaf'])
    print("Jaccard Similarity\n", jaccard_similarity_df)
    euclidean_distance_matrix = euclidean_distances(tv_matrix)
    euclidean_distance_df = pd.DataFrame(euclidean_distance_matrix, columns=['harry_potter', 'lotr', 'asoiaf'], index=['harry_potter', 'lotr', 'asoiaf'])
    print("Euclidean Similarity\n", euclidean_distance_df)
    # cluster analysis
    z = linkage(cosine_similarity_matrix, 'ward')
    pd.DataFrame(z, columns=['Document\Cluster 1', 'Document\Cluster 2', 'Distance', 'Cluster Size'], dtype='object')
    plt.figure(figsize=(8, 3))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data point')
    plt.ylabel('Distance')
    dendrogram(z, labels=['harry_potter', 'lotr', 'asoiaf'])
    plt.axhline(y=1.0, c='k', ls='--', lw=0.5)
    plt.show()


if __name__ == '__main__':
    main()
