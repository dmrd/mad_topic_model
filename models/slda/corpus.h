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
        int label;
        int wordTypes; 

        document();
        document(int * len);   
        document(int * len, int wt);
        ~document();
    private:
        void init(int * len, int wt);
};

document::document(int * len)
{
    init(len,1);
}

document::document(int* len, int  wt)
{
    init(len,wt);
}
void document::init(int * len, int wt)
{
        length = len;
        wordTypes  = wt;

        words = new int *  [wordTypes];
        counts = new int * [wordTypes];

        for (int t = 0; t < wordTypes; t++)
        {
            words[t] = new int [length[t]];
            counts[t] = new int [length[t]];
        }

        total = new int[wordTypes];
        label = -1;
}

document::~document()
    {
        if (words != NULL)
        {
            for (int t = 0; t < wordTypes; t++)
            {
                delete [] words[t];
                delete [] counts[t];
            }
            delete [] words;
            delete [] counts;

            delete[] length;
            delete [] total;
            label = -1;
        }
    }

document::document()
{
    words = NULL;
    counts = NULL;
    length = NULL;
    total = NULL;
    label = -1;
}



class corpus
{
public:
    corpus();
    ~corpus();
    void read_data(const char * data_filename, const char * label_filename);
    int * max_corpus_length();
public:
    int num_docs;
    int * size_vocab;
    int num_classes;
    int * num_total_words;
    int num_word_types;
    vector<document*> docs;
};

#endif // CORPUS_H
