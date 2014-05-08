import argparse
from yaml import load


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert SLDA output to Termite format.')
    parser.add_argument(
        'input_file', help='input filename (output by SLDA model)')
    parser.add_argument('dictionaries', metavar='dictionary', nargs='+',
                        help='index-to-term dictionary, one for each word type')
    args = parser.parse_args()
    data = load(file(args.input_file).read())

    for word_type in range(data['number_of_word_types']):
        dictionary = args.dictionaries[word_type]
        dictionary = file(dictionary).read().split('\n')

        results = data['metrics'][word_type]

        # Topic Index
        topic_index = file('topic-index' + str(word_type) + '.txt', 'w+')
        for i in range(results['number_of_topics']):
            topic_index.write(str(i) + '\n')

        # Term Index
        term_index = file('term-index' + str(word_type) + '.txt', 'w+')
        for i in range(results['size_of_vocab']):
            term_index.write(dictionary[i] + '\n')

        # Term-Topic Matrix
        term_topic = file('term-topic-matrix' + str(word_type) + '.txt', 'w+')
        for i in range(results['number_of_topics']):
            for j in range(results['size_of_vocab']):
                n = results['vocab_distribution'][i][j]
                term_topic.write(str(n) + ' ')
            term_topic.write('\n')
