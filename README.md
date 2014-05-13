
# MAD Topic Model

Wordless authorship recognition using topic models.

## Dependencies

See requirements.txt for Python dependencies.

## Data

The `data` folder contains the final scraped datasets used in the paper.

For the scrapers' source code, refer to the `scrapers` folder. READMEs in each of the subfolders explain how to begin the programs. Project Gutenberg texts were downloaded manually because of the low number of authors needed for that portion of the project.

The `slda_input_files` folder contains files in a generalized format ready to be read into the SLDA model. Input files have been generated for several combinations of word types and values of $n$. These input files are generated with `pipeline.py`.


### NLTK Datasets

- cmudict
- punkt
- maxent_treebank_pos_tagger

### gsl library for c++

## Resources

### Models

The MAD model is implemeneted in models/slda. Our implementation Chong Wang's sLDA implementation at http://www.cs.cmu.edu/~chongw/slda/. The algorithm is mostly coded in slda.cpp, with help from opt.cpp and dirichlet.c for computing difficult gradients. settings.h contains a (for the time being) rather disorganized list of settings, allowing for different inference schemes (variational, stochastic, and the orginal fixed point/variational combination), regularization - L1 (not fully tested. uses http://www.chokkan.org/software/liblbfgs/), L2, and smoothing for the Dirichlet MLE and per-topic vocabulary distributions. settings.h also controls the number of iterations the algorithm should run. For some reason, the settings.h is having difficulty reading the settings.txt input file, so it is best to hard code the desired settings into settings.h. Another README is provided within models/slda explaining how to run the software from command line. 
