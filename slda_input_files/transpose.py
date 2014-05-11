import sys

data = file(sys.argv[1]).read()
data = data.strip().split('\n')
data = map(lambda e: e.split(' '), data)
data = [list(i) for i in zip(*data)]

for line in data:
    print '\t'.join(line)
