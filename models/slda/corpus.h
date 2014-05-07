// (C) Copyright 2009, Chong Wang, David Blei and Li Fei-Fei

// written by Chong Wang, chongw@cs.princeton.edu

// This file is part of slda.

// slda is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// slda is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#ifndef CORPUS_H
#define CORPUS_H

#include <vector>
using namespace std;

class document
{
    public:
        int ** words;
        int ** counts;
        int *length;
        int *total;
        int label; // label is the author
        int wordTypes;  // word types is the type of n gram

        document();
        document(int wt);
        void set_length(int t, int len);

        document(int * len, int wt);
        ~document();
    private:
        void init(int * len, int wt);
};


class corpus
{
public:
    corpus();
    corpus(int T);
    ~corpus();
    void read_data(const char * data_filename0, const char * label_filename);
    int * max_corpus_length();
public:
    int num_docs; // number documents
    int * size_vocab; // size of vocab for each type
    int num_classes; // number of authors
    int * num_total_words;
    int num_word_types;
    vector<document*> docs;
};

#endif // CORPUS_H
