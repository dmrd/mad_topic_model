import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert SLDA output to Termite format.')
    parser.add_argument('model_file')
    parser.add_argument('term_file')
    args = parser.parse_args()

    terms = file(args.term_file).read().split('\n')
    i = 0

    for line in file(args.model_file).readlines():
        data = line.split(' ')[1:]

        print str(i) + '\t',

        for entry in data:
            entry = entry.split(':')
            term_index = int(entry[0])
            term = str(terms[term_index]).replace(' ', '')
            term_count = int(entry[1])
            print ' '.join([term] * term_count),
        print '\n',
        i += 1
