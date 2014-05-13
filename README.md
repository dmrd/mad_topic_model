
# MAD Topic Model

Wordless authorship recognition using topic models. MAD implements a multivalent supervised Latent Dirichlet Allocation algorithm that operates on vocabularies of n-gram stylistic and stylometric features, such as part-of-speech tags and syllable counts. MAD can be used for both author classification and exploratory analysis as the generated topic models reveal hidden structure among authors' writing styles.

## Dependencies

See requirements.txt for Python dependencies.

In addition, the following NLTK libraries are required:

- cmudict
- punkt
- maxent_treebank_pos_tagger

The GSL library for C++ is required.

## Data

The `data` folder contains the final scraped datasets used in the paper.

For the scrapers' source code, refer to the `scrapers` folder. READMEs in each of the subfolders explain how to begin the programs. Project Gutenberg texts were downloaded manually because of the low number of authors needed for that portion of the project.

The `slda_input_files` folder contains files in a generalized format ready to be read into the SLDA model. Input files have been generated for several combinations of word types and values of $n$. These input files are generated with `pipeline.py`.

## Models

The MAD model is implemented in `models/slda`. Our implementation is based on that of [Chong Wang](http://www.cs.cmu.edu/~chongw/slda/). The algorithm is mostly contained in `slda.cpp`, with help from `opt.cpp` and `dirichlet.c` for computing difficult gradients.

`settings.h` contains a list of settings, allowing for different inference schemes (variational, stochastic, and the original fixed point-variational combination), regularization (including L1 (not fully tested, but based on [liblbfgs](http://www.chokkan.org/software/liblbfgs/)), L2, and smoothing for the Dirichlet MLE) and per-topic vocabulary distributions. `settings.h` also controls the number of iterations the algorithm should run. `settings.h` has difficulty reading the settings.txt input file, so it is best to hard code the desired settings into `settings.h`. A supplemental README is provided within `models/slda` with information on how to run the software from command line.

## Feature Extraction

Python modules are provided for extracting the necessary stylistic features from text, such as part-of-speech tags and syllable counts. The majority of the feature extraction functions can be found in `features/analyzer.py`. In addition, a novel technique for extracting meter 8-grams is provided in `features/meter.py`. To allow for further analysis (after n-gram generation), `features/extract.py` provides functions for identifying text snippets within documents that match the given stylistic n-gram.