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

#ifndef SLDA_H
#define SLDA_H
#include "settings.h"
#include "corpus.h"

typedef struct {
    double * z_bar_m;
    double * z_bar_var;
} z_stat;

typedef struct {
    double ** word_ss; /* indexed by document, then word type, the word */
    double * word_total_ss; /* indexed by word type, then topic*/

    z_stat ** z_bar /* one per word topic, one per document,*/;
    int * labels;
    int * tot_labels;
} suffstats;


typedef struct {
    double * alpha_1; // the parameter for the dirichlet, indexed by word type
    // ammenable to optimization
    double * alpha_2; // the parameter for the dirichlet, index by word type
    // fixed to optimization
    double * alpha_t;

    double alpha_sum_1;
    double alpha_sum_2;
    double alpha_sum_t;
} alphas;


class slda
{
public:
    slda();
    ~slda();
    void free_model();
    void init(double alpha_, int num_topics_, const corpus * c);
    void v_em(corpus * c, const settings * setting,
              const char * start, const char * directory);

    void save_model(const char * filename);
    void save_model_text(const char * filename);
    void load_model(const char * model_filename);
    void infer_only(corpus * c, const settings * setting,
                    const char * directory);

    suffstats * new_suffstats(int num_docs);
    void free_suffstats(suffstats * ss);
    void zero_initialize_ss(suffstats * ss);
    void random_initialize_ss(suffstats * ss, corpus * c);
    void corpus_initialize_ss(suffstats* ss, corpus * c);
    void load_model_initialize_ss(suffstats* ss, corpus * c);
    void mle(suffstats * ss, int eta_update, const settings * setting);

    double doc_e_step(document* doc, double* gamma, double** phi, suffstats * ss, int eta_update, const settings * setting);

    double lda_inference(document* doc, double* var_gamma, double** phi, const settings * setting);
    double lda_compute_likelihood(document* doc, double** phi, double* var_gamma);
    double slda_inference(document* doc, double* var_gamma, double** phi, const settings * setting);
    double slda_compute_likelihood(document* doc, double** phi, double* var_gamma);

    void save_gamma(char* filename, double** gamma, int num_docs);
    void write_word_assignment(FILE* f, document* doc, double** phi);


public:
    
    double * scaling; // scales prior to match author prolificness
   
    int num_docs; /* number of documents*/
    int * docs_per; // # documents per author, indexed by author 


    int * num_topics; // # of topics, indexed by word type
    int num_classes; // number of authors
    int * size_vocab; // size of vocab for each type of word

    int num_word_types;

    double *** log_prob_w; //the log of the topic distribution
    // indexed by word type, then topic, then word
    double *** eta; //softmax regression, in general, there are num_classes-1 etas, we don't need a intercept here, since \sum_i \bar{z_i} = 1
    // indexed first by class type, then by word type, then by word
};

#endif // SLDA_H

