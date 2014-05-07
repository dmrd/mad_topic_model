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

#include "corpus.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sstream>
#include <iostream>

// constructs a document with wt wordtypes
// and length array len
document::document(int* len, int  wt)
{
    init(len,wt);
}

// construcs a document with wt wordtypes
document::document(int wt)
{
    wordTypes = wt;
    words = new int *  [wordTypes];
    counts = new int * [wordTypes];
    total = new int[wordTypes];
    label = -1;

}

// sense length of document in t^th word time
void document::set_length(int t, int len)
{
    words[t] = new int [length[t]];
    counts[t] = new int [length[t]];
    length[t] = len;
    total[t] = 0;
}

// helper method for construction
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

corpus::corpus()
{
    num_docs = 0;
    size_vocab = NULL;
    num_classes = 0;
    num_total_words = NULL;
    num_word_types = 1;
}

corpus::corpus(int T)
{
    num_word_types = T;
    num_docs = 0;
    size_vocab = new int [num_word_types];
    num_classes = 0;
    num_total_words = new int [num_word_types];
    for (int t = 0; t < T; t ++)
        num_total_words[t] = 0;
}


corpus::~corpus()
{
    for (int i = 0; i < num_docs; i ++)
    {
        document * doc = docs[i];
        delete doc;
    }
    docs.clear();

    num_docs = 0;
    delete [] size_vocab;
    num_classes = 0;
    delete [] num_total_words;
}

void corpus::read_data(const char * data_filename0,
                       const char * label_filename)
{
    int OFFSET = 0;
    int length = 0, count = 0, word = 0,
        n = 0, nd = 0, label = -1, t =0;

    nd = 0;
    int * nw = new int [num_word_types];
    const char ** data_filename = new const char * [num_word_types];

    for (t = 0; t < num_word_types; t++)
    {
        char * app;
        sprintf(app, "_%i", t);
        std::string s1 = std::string(data_filename0);
        std::string s2 = std::string(app);

        data_filename[t] = (s1+s2).c_str();
    }

    FILE * fileptr;

    for (t = 0; t < num_word_types; t ++)
    {
        nw[t] = 0;
        fileptr = fopen(data_filename[t], "r");
        printf("\nreading data from %s\n", data_filename[t]);

        while ((fscanf(fileptr, "%10d", &length) != EOF))
        {
            document * doc;
            if (t == 0)
                 doc = new document(t);
            else
                 doc = docs[nd];

            doc->set_length(t,length);

            for (n = 0; n < length; n++)
            {
                fscanf(fileptr, "%10d:%10d", &word, &count);
                word = word - OFFSET;
                doc->words[t][n] = word;
                doc->counts[t][n] = count;
                doc->total[t] += count;
                if (word >= nw[t])
                {
                    nw[t] = word + 1;
                }
            }
            num_total_words[t] += doc->total[t];
            docs.push_back(doc);
            nd++;
        }
        fclose(fileptr);
    }
    num_docs = nd;
    size_vocab = nw;
    printf("number of docs  : %d\n", nd);

    for (t = 0; t<num_word_types; t++)
    {
        printf("word type %i\n", t);
        printf("number of terms : %d\n", nw[t]);
        printf("number of total words : %d\n", num_total_words[t]);
    }

    fileptr = fopen(label_filename, "r");
    printf("\nreading labels from %s\n", label_filename);
    nd = 0;
    while ((fscanf(fileptr, "%10d", &label) != EOF))
    {
        document * doc = docs[nd];
        doc->label = label;
        if (label >= num_classes)
        {
            num_classes = label + 1;
        }
        nd ++;
    }
    assert(nd == int(docs.size()));
    printf("number of classes : %d\n\n", num_classes);
}

int * corpus::max_corpus_length() {
    int * max_length = new int [num_word_types];
    for (int t = 0; t < num_word_types; t++)
        max_length[t] =0;

    for (int d = 0; d < num_docs; d++) {
         for (int t = 0; t < num_word_types; t++)
        {
            if (docs[d]->length[t] > max_length[t])
                max_length[t] = docs[d]->length[t];
        }
    }
    return max_length;
}
