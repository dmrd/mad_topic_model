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
