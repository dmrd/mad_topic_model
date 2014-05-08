from analyzer import stress_counts_by_syllable


def meter_ngrams(text):
    stress_counts = stress_counts_by_syllable(text, SECONDARY=False)

    since_start = 0
    since_comma = 0

    ngrams = []
    current = []
    for (stresses, punc) in stress_counts:
        for stress in stresses:

            if len(current) >= 6:
                current.pop(0)

            current.append(stress)

            if len(current) == 6:
                ngrams.append(tuple(current + [since_start, since_comma]))
                since_start = (since_start + 1) % 2
                since_comma = (since_comma + 1) % 2

        if punc == ',':
            since_comma = 0
        else:

            if len(current) < 6:
                current += [-1] * (6 - len(current))
                ngrams.append(tuple(current + [since_start, since_comma]))

            since_start = 0
            current = []

    return ngrams
