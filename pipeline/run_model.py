#!/usr/bin/python
# Usage:
# run_model.py data_name min_docs_per_author nfolds ntopics
# data_name is gutenberg/nweekly/quora
#     Corresponds to files in slda_input_files
#     Outputs files in ./output
# min_docs_per_author: Doc number threshold to include an author
#
# This script makes me feel dirty inside D= , but it works =D
import os
import sys
import subprocess

try:
    data_name = sys.argv[1]
    min_doc_per_author = int(sys.argv[2])
    nfolds = int(sys.argv[3])
    ntopics = int(sys.argv[4])
except:
    print("Usage: run_model.py data_name min_docs_per_author nfolds ntopics")
    exit()

NTYPES = 6

#os.system("make -C ../models/slda clean && make -C ../models/slda")
os.system("mkdir output/{n} output/{n}/models output/{n}/data".format(n=data_name))
os.system("rm -f output/{n}/models/* output/{n}/data/*".format(n=data_name))
os.system("sh copy_ngrams.sh ../slda_input_files/{n} output/{n}/data/{n} {min_doc}".format(n=data_name,
                                      min_doc=min_doc_per_author))
os.system("python test_train_split.py output/{n}/data/{n} {types} {nfolds}".format(n=data_name, types=NTYPES, nfolds=nfolds))
print("python test_train_split.py output/{n}/data/{n} {types} {nfolds}".format(n=data_name, types=NTYPES, nfolds=nfolds))

slda_est = []
slda_inf = []

# tt: test/train
# n: name (quora/gutenberg/etc)
# fold: current fold
data_loc = "./output/{n}/data/fold{fold}_{tt}_{n}"
label_loc = "./output/{n}/data/fold{fold}_{tt}_{n}_labels"
model_loc = "./output/{n}/models/{fold}/"

slda_est.append("../models/slda/slda est {types}")
slda_est.extend([data_loc, label_loc])
slda_est.append("../models/slda/settings.txt 0.1 random")
slda_est.append(model_loc)
slda_est.append(' '.join([str(ntopics) for _ in range(NTYPES)]))

slda_inf.append("../models/slda/slda inf {types}")
slda_inf.extend([data_loc, label_loc])
slda_inf.append("../models/slda/settings.txt")
slda_inf.append(model_loc + "final.model")
slda_inf.append(model_loc)


for fold in range(nfolds):
    print("********************* Running fold {} *********************".format(fold))
    # Make output folder
    os.system("mkdir output/{n}/models/{fold}".format(n=data_name, fold=fold))

    print("**************** Training ****************")
    estimate = ' '.join(slda_est).format(n=data_name, tt='train', fold=fold, types=NTYPES)
    print(estimate)
    os.system(estimate)

    print("**************** Testing ****************")
    infer = ' '.join(slda_inf).format(n=data_name, tt='test', fold=fold, types=NTYPES)
    print(infer)
    os.system(infer)


print("\n\n\n**************** RESULTS ****************\n")
log = []

for fold in range(nfolds):
    log.append("\n**** Fold {} results ****".format(fold))
    log.append(subprocess.check_output("tail -n5 output/{n}/models/{fold}/inf-labels.dat".format(n=data_name,
                                                                                                 fold=fold), shell=True))
with open("logs/{}_{}_{}_{}.log".format(data_name, min_doc_per_author,
                                        nfolds, ntopics), 'w') as f:
    f.write('\n'.join(log))
print('\n'.join(log))
