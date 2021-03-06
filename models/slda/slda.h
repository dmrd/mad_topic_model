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
#include <gsl/gsl_matrix.h>

#include <vector>
using namespace std;

typedef struct {
    double * z_bar_m;
    double * z_bar_var;
} z_stat;

typedef struct {
    vector<vector<double> > word_ss; /* indexed by document, the word */
    vector<double> word_total_ss; /* indexed by word type, then topic */
    //double ** word_ss;
    //double * word_total_ss;

    z_stat * z_bar /* per document */;
    int * labels;
    int * tot_labels;
} suffstats;

// stores digamma values which are costly to compute
typedef struct 
{
 vector<double> digamma_vec;
 double digamma_sum;
} digammas;

typedef std::pair<double,int> mypair;


typedef struct {
    double * alpha_1; // the parameter for the dirichlet, indexed by word type
    // ammenable to optimization
    double * alpha_2; // the parameter for the dirichlet, index by word type
    // fixed to optimization
    double * alpha_t;

    double * phi_total;

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
    void init(double epsilon2, int * num_topics_, const corpus * c);
    void v_em(corpus * c, const settings * setting,
              const char * start, const char * directory);
    void updatePrior(double *** var_gamma, const settings * setting, corpus * c);
    void globalPrior(double *** var_gamma, const settings * setting);
    void fitDirichlet(gsl_matrix * mat);
    void updatePriorSimple(double *** var_gamma, const settings * s);


    void save_model(const char * filename);
    void save_model_text(const char * filename);
    void load_model(const char * filename);


    void infer_only(corpus * c, const settings * setting,
                    const char * directory);

    suffstats * new_suffstats( int t);
    //void free_suffstats(suffstats ** ss, int t);
    void zero_initialize_ss(suffstats * ss, int t);
    //for svi
    void zero_initialize_word(suffstats * ss, int t);
    void random_initialize_ss(suffstats * ss, corpus * c, int t);
    void corpus_initialize_ss(suffstats * ss, corpus * c, int t);
    void load_model_initialize_ss(suffstats* ss, corpus * c, int t);

    // mle global computes the dictionary updates
    // mle logistic computes logistic repression updates
    void mle_logistic( std::vector<suffstats *> ss, double rho, 
        const settings * setting, std::vector<int> docs);
    void mle_global(vector<suffstats *> ss, double rho, const settings * setting);

    void infer_only_2(corpus * c, const settings * setting, const char * directory);
    double doc_e_step(document* doc, double* gamma, double** phi, suffstats * ss,
     int eta_update, int _docNum, int t, const settings * setting);

    double slda_inference_1(document* doc, double ** var_gamma, double *** phi,
                            alphas *** as, int d, const settings * setting);
    double lda_inference(document* doc, double* var_gamma, double** phi, const settings * setting, int t, int a);
    double lda_compute_likelihood(document* doc, double** phi, double* var_gamma, int t, int a);
    double slda_inference(document* doc, double** var_gamma, double*** phi,
      alphas *** as, int d, const settings * setting);
    double slda_compute_likelihood(document* doc, double*** phi, double** var_gamma, int d);
    double doc_perplexity(document* doc, double ** expAlpha, double *** phi);
    void save_gamma(char* filename, double*** gamma, int num_docs);
    void write_word_assignment(FILE* f, document* doc, double*** phi);
    void init_alpha(double epsilon2);
    void init_global_as(double epsilon2);
    void init_dg(int num_docs);

    //void updatePrior();

    int vec_index(int t, int l, int k);
    //int vec_index2(int t, int l, int k, vector<int> stoch_authors);

    int getDoc(int a, int d);
    double add_penalty(const settings * setting);
    double get_rho(int i);

    //stochastic optimization
    //void stoch_logistic(vector<suffstats *> ss, const settings * setting,
    //    std::vector<double> author_prob, int author_trials,
    //    std::vector<double> doc_prob, int doc_trials);
   
    double doc_e_step(document* doc, double* gamma, double** phi,
                        suffstats * ss, int eta_update, int _docNum, int t,
                        double scaling, const settings * setting);


    static bool mComp ( const mypair& l, const mypair& r);
        
public:

    //double * scaling; // scales prior to match author prolificness

    int num_docs; /* number of documents*/
    int * docs_per; // # documents per author, indexed by author
    std::vector<std::vector<int> > docAuthors;

    double epsilon;
    int * num_topics; // # of topics, indexed by word type
    int num_classes; // number of authors
    int * size_vocab; // size of vocab for each type of word

    int num_word_types;

    //the log of the topic distribution
    vector< vector < vector < double > > >  log_prob_w;
    vector< vector < vector < double > > >  lambda;


    // indexed by word type, then topic, then word
    // softmax regression, in general, there are num_classes-1 etas, we
    // don't need a intercept here, since \sum_i \bar{z_i} = 1
    vector< vector < vector < double > > >  eta;

    // indexed first by class type, then by word type, then by word
    alphas *** as;
    alphas ** as_global;
    digammas *** dg;

    int first_run; //different instructions for first run of stochastic 
    // gradient descent
    double top_prior;


};

#endif // SLDA_H
