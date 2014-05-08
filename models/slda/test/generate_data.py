"""
    Generates artificial data to test SLDA model
    Outputs:
    + n_topics files starting with given prefix:
        prefix_1
        prefix_2
        prefix_...
        prefix_n
      where n = n_topics
    + a label file:
        prefix_label
"""

import argparse
import numpy as np
from collections import defaultdict

def generate_documents(n_authors, n_topics, n_docs, n_words):
    """
    Implements generative process for LDA

    Generates n_docs for each of the n_authors, where each author has
    a distribution over n_topics and each topic has a distribution
    over n_words
    """

    # Generate author dirichlet distributions over topics
    author_p = []
    for _ in range(n_authors):
        x = np.random.rand(n_topics)
        author_p.append(x)

    # Generate topic multinomial distributions over words
    # (drawn from dirichlet)
    x = np.random.rand(n_words)
    topic_p = np.random.dirichlet(x, n_topics)

    # docs is a list of [lists of documents]
    # docs[i] is a list of documents by author i
    # Each document is a dictionary from {word id: count}
    docs = []
    for a in range(n_authors):
        # +a to get unevent number of documents per author
        for d in range(n_docs + a):
            doc = defaultdict(lambda: 0)
            doc['AUTHOR'] = a
            word_counts = np.zeros(n_words)
            words_in_doc = np.random.poisson(10)

            # This documents multinomial distribution over topics
            doc_topic_dist = np.random.dirichlet(author_p[a], words_in_doc)
            # Number of words from each topic in this document
            doc_topics = np.array([np.random.multinomial(1, dist)
                                  for dist in doc_topic_dist]).sum(axis=0)

            for topic, count in enumerate(doc_topics):
                for i in range(count):
                    word = np.random.multinomial(1, topic_p[topic]).nonzero()[0][0]
                    doc[word] += 1
            docs.append(doc)
    return docs


def save_docs(docs, prefix, topic_type):
    """
    docs: list of length #authors where docs[i] = [list of {docs}]
    as produced by generate_documents

    prefix: Output prefix.  Output file is $(prefix)_$(topic_type)
    topic_type: word type id (integer)
    """
    with open(prefix + "_" + str(topic_type), 'w') as f:
        for doc in docs:
            f.write(str(len(doc)) + " ")
            for k in doc:
                f.write("{}:{} ".format(k, doc[k]))
            f.write("\n");

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate artificial data to test SLDA model')
    parser.add_argument('--prefix', help='Prefix for output files')
    parser.add_argument('--n_authors', help='Number of authors to generate',
                         default=2, type=int)
    parser.add_argument('--n_docs',
                        help='Number of documents per author to generate',
                        default=20, type=int)
    parser.add_argument('--n_topics',
                        help='Number of topics',
                        default=2, type=int)
    parser.add_argument('--n_types',
                        help='Number of ngram types',
                        default=3, type=int)
    parser.add_argument('--n_words',
                        help='vocabulary size',
                        default=1500, type=int)
    args = parser.parse_args()

    # Write a file for each ngram type
    type_docs = None
    for i in range(args.n_types):
        type_docs = generate_documents(args.n_authors, args.n_topics,
                                       args.n_docs, args.n_words)
        save_docs(type_docs, args.prefix, i)

    # Go through once and write the label file
    with open(args.prefix + "_labels", 'w') as f:
        for doc in type_docs:
            f.write(str(doc['AUTHOR']) + "\n")
