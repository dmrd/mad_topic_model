#!/bin/sh
# Sanity check on model - generates trivial data and trains/fits on test=train set
# Should get near 1.0 accuracy
make -C .. clean
make -C ..
mkdir sanity_check sc
python generate_data.py --prefix sanity_check/sc --n_topics 4 --n_authors 4
../slda est 3 ./sanity_check/sc ./sanity_check/sc_labels ../settings.txt 0.1 random ./sc 4 4 4
../slda inf 3 ./sanity_check/sc ./sanity_check/sc_labels ../settings.txt ./sc/final.model ./sc
