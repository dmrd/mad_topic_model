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

#include "slda.h"
#include <time.h>
#include <iostream>
#include "utils.h"
#include "assert.h"
#include "opt.h"
#include "dirichlet.h"
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <vector>
//using namespace std;


const int NUM_INIT = 50;
const int LAG = 10;
const int LDA_INIT_MAX = 0;
const int MSTEP_MAX_ITER = 50;

slda::slda()
{
    //ctor
    epsilon = .2;
    num_topics = 0;
    num_classes = 0;
    size_vocab = 0;
}

slda::~slda()
{
    free_model();
}

/*
 * initializes the values of alpha in the model
 */
void slda::init_alpha(double epsilon2)
{
    as = new alphas ** [num_word_types];

    for (size_t t = 0; t < num_word_types; t++)
    {
        as[t] = new alphas *  [num_classes];

        int KK = num_topics[t];

        for (size_t a = 0; a < num_classes; a++)
        {
            as[t][a] = new alphas;
            as[t][a]->alpha_sum_1 = KK*epsilon2;
            as[t][a]->alpha_sum_2 = KK*epsilon2;
            as[t][a]->alpha_sum_t = 2*KK*epsilon2;

            as[t][a]->alpha_1 = new double [KK];
            as[t][a]->alpha_2 = new double [KK];
            as[t][a]->alpha_t = new double [KK];
            for (int k = 0; k < KK; k++)
            {
                as[t][a]->alpha_1[k] = epsilon2;
                as[t][a]->alpha_2[k] = epsilon2;
                as[t][a]->alpha_t[k] = 2*epsilon2;
            }
        }
    }
}

/*
 * Initialize per author dirichlet alphas
 */
void slda::init_global_as(double epsilon2)
{
    as_global = new alphas * [num_word_types];

    for (size_t t = 0; t < num_word_types; t++)
    {
        as_global[t] = new alphas;

        int KK = num_topics[t];
        as_global[t]->alpha_sum_1 = KK*epsilon2;
        as_global[t]->alpha_sum_2 = KK*epsilon2;
        as_global[t]->alpha_sum_t = 2*KK*epsilon2;

        as_global[t]->alpha_1 = new double [KK];
        as_global[t]->alpha_2 = new double [KK];
        as_global[t]->alpha_t = new double [KK];
        for (int k = 0; k < KK; k++)
        {
            as_global[t]->alpha_1[k] = epsilon2;
            as_global[t]->alpha_2[k] = epsilon2;
            as_global[t]->alpha_t[k] = 2*epsilon2;
        }
    }
}

void slda::init_dg(int num_docs)
{
    dg = new digammas ** [num_word_types];
    for (size_t t = 0; t < num_word_types; t++)
    {
        dg[t] = new digammas * [num_docs];
        for (size_t d = 0; d < num_docs; d ++)
        {
            dg[t][d] = new digammas;
            for (size_t k = 0; k < num_topics[t]; k ++)
            {
                dg[t][d]->digamma_vec.push_back(0);
            }
        }

    }
}

void slda::init(double epsilon2, int * num_topics_,
                const corpus * c)
{
    num_topics = num_topics_;
    top_prior = 1;
    first_run = 1;

    docs_per = c->docsPer;
    docAuthors = c->docAuthors;

    num_word_types = c->num_word_types;
    num_docs = c->num_docs;

    size_vocab = c->size_vocab;
    num_classes = c->num_classes;
    epsilon = epsilon2;


    init_alpha(epsilon2);
    init_global_as(epsilon2);
    init_dg(c->num_docs);

    // iterate through each type of word
    for (int t = 0; t < num_word_types; t++)
    {
        lambda.push_back(std::vector<std::vector<double > >());
        log_prob_w.push_back(std::vector<std::vector<double > >());
        eta.push_back(std::vector<std::vector<double > >());

        // go through each topic for a word type
        for (int k = 0; k < num_topics[t]; k++)
        {
            log_prob_w[t].push_back(std::vector<double>(size_vocab[t], 0));
            lambda[t].push_back(std::vector<double>(size_vocab[t], 0));
        }

        for (int i = 0; i < num_classes-1; i++)
        {
            eta[t].push_back(std::vector<double>(num_topics[t], 0));
        }
    }
}

/*
 * free the model
 */

void slda::free_model()
{
    /* TODO: WRITE FREE TO WORK WITH VECTORS */
    // if (log_prob_w != NULL)
    // {
    //     for (int t = 0; t < num_word_types; t++)
    //     {
    //         for (int k = 0; k < num_topics[t]; k++)
    //             delete [] log_prob_w[t][k];
    //         delete [] log_prob_w[t];
    //     }
    //     delete [] log_prob_w;
    //     log_prob_w = NULL;
    // }
    // if (eta != NULL)
    // {
    //     for (int t = 0; t < num_word_types; t++)
    //     {
    //         for (int i = 0; i < num_classes-1; i ++)
    //             delete [] eta[t][i];
    //         delete [] eta[t];
    //     }
    //     delete [] eta;
    //     eta = NULL;
    // }
}


/*
 * Create the data structure for sufficient statistic
 * for a given word type
 * @param t is the word type
 * @param a is the corresponding author
 */

suffstats * slda::new_suffstats(int t)
{
    suffstats * ss = new suffstats;

    std::vector<double>::iterator it;
    it = ss->word_total_ss.begin();
    for (int i = 0; i < num_topics[t]; i++)
        ss->word_total_ss.push_back(0);

    //ss->word_total_ss.insert(it,num_topics[t],0);
    //ss->word_ss = new double * [num_topics[t]];

    for (int k = 0; k < num_topics[t]; k++)
    {
        //ss->word_ss[k] = new double [size_vocab[t]];
        //memset(ss->word_ss[k], 0, sizeof(double)*size_vocab[t]);
        ss->word_ss.push_back(std::vector<double>(size_vocab[t],0));
    }

    int num_var_entries = num_topics[t]*(num_topics[t]+1)/2;
    ss->z_bar = new z_stat [num_docs];
    for (int d = 0; d < num_docs; d++)
    {
        ss->z_bar[d].z_bar_m = new double [num_topics[t]];
        ss->z_bar[d].z_bar_var = new double [num_var_entries];
        memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics[t]);
        memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);

    }

    ss->labels = new int [num_docs];
    memset(ss->labels, 0, sizeof(int)*(num_docs));
    ss->tot_labels = new int [num_classes];
    memset(ss->tot_labels, 0, sizeof(int)*(num_classes));

    return(ss);
}


void slda::zero_initialize_word(suffstats * ss, int t)
{
    //memset(ss->word_total_ss, 0, sizeof(double)*num_topics[t]);
    ss->word_total_ss.assign(num_topics[t],0);
    for (int k = 0; k < num_topics[t]; k++)
    {
        ss->word_ss[k].assign(size_vocab[t],0);
        //memset(ss->word_ss[k], 0, sizeof(double)*size_vocab[t]);
    }
}

/*
 * initialize the sufficient statistics with zeros
 */

void slda::zero_initialize_ss(suffstats * ss, int t)
{
    //memset(ss->word_total_ss, 0, sizeof(double)*num_topics[t]);
    ss->word_total_ss.assign(num_topics[t],0);
    for (int k = 0; k < num_topics[t]; k++)
    {
        ss->word_ss[k].assign(size_vocab[t],0);
        //memset(ss->word_ss[k], 0, sizeof(double)*size_vocab[t]);
    }

    int num_var_entries = num_topics[t]*(num_topics[t]+1)/2;

    for (int d = 0; d < num_docs; d++)
    {
        memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics[t]);
        memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);
    }

    //ss->num_docs = 0;
}


/**
 * returns the document index of
   author a's d^{th} document
 **/
int slda::getDoc(int a, int d)
{
    return docAuthors[a][d]; //corpus.getDoc(a,d)
}

/*
 * initialize the sufficient statistics with random numbers
 */
void slda::random_initialize_ss(suffstats * ss, corpus* c, int t)
{

    gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
    time_t seed;
    time(&seed);
    gsl_rng_set(rng, (long) seed);
    int k, w, d, j, idx;


    for (k = 0; k < num_topics[t]; k++)
    {
        for (w = 0; w < size_vocab[t]; w++)
        {
            ss->word_ss[k][w] = 1.0/size_vocab[t] + 0.1*gsl_rng_uniform(rng);
            ss->word_total_ss[k] += ss->word_ss[k][w];
        }
    }

    for (d = 0; d < num_docs; d++)
    {
        document * doc = c->docs[d];
        ss->labels[d] = doc->label; // in general, doc label = author
        ss->tot_labels[doc->label]++;

        double total = 0.0;
        for (k = 0; k < num_topics[t]; k++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }
        for (k = 0; k < num_topics[t]; k++)
        {
            ss->z_bar[d].z_bar_m[k] /= total;
        }
        for (k = 0; k < num_topics[t]; k++)
        {
            for (j = k; j < num_topics[t]; j++)
            {
                idx = map_idx(k, j, num_topics[t]);
                if (j == k)
                    ss->z_bar[d].z_bar_var[idx] = ss->z_bar[d].z_bar_m[k] / (double)(doc->total[t]);
                else
                    ss->z_bar[d].z_bar_var[idx] = 0.0;

                ss->z_bar[d].z_bar_var[idx] -=
                    ss->z_bar[d].z_bar_m[k] * ss->z_bar[d].z_bar_m[j] / (double)(doc->total[t]);
            }
        }
    }

    gsl_rng_free(rng);
}


/**
   Optimizes Per Author Dirichlet Prior
 **/
void slda::updatePrior(double *** var_gamma, const settings * s, corpus * c)
{
    double isSmoothed = s->IS_SMOOTHED;
    double smoothWeight = s->SMOOTH_WEIGHT;

    for (int t = 0; t < num_word_types; t++)
    {
        
        // used for smoothing
        /**
        gsl_vector * w;
        if (isSmoothed)
        {
            w = gsl_vector_alloc(num_topics[t]);
            for (int k = 0; k < num_topics[t]; k++)
                gsl_vector_set(w, k, 1/((double)(num_topics[t])));
        }
        **/
        double ** vals = new double *[num_classes];
       // cout << "t: " << t;
        for (int a = 0; a < num_classes; a++ )
        {
            vals[a]  = new double[num_topics[t]];
            for (int k = 0; k < num_topics[t]; k++)
            {
                vals[a][k] = 0;
            }            

        }

        for (int d =0; d < num_docs; d++)
        {
            for (int k = 0; k < num_topics[t]; k++)
                vals[c->docs[d]->label][k] += var_gamma[d][t][k];
        }

        for (int a = 0; a < num_classes; a++ )
        {
            for (int k = 0; k < num_topics[t]; k++)
            {
                //cout << "vals " << a << " , " << k << ": " << vals[a][k] << "\n";
            }            

        }


        /**
            cout << "author " << a << "\n";
            gsl_matrix * mat = gsl_matrix_alloc(docs_per[a],num_topics[t]);
            gsl_vector * v;

            for (int dd = 0; dd < docs_per[a]; dd++)
            {
                int d = getDoc(a,dd);
                cout << "document " << d << "\n";
                cout << "document dg" <<  << "\n";



                // digamma of sum
                double digsum = dg[t][d]->digamma_sum;
                std::vector<double> dig = dg[t][d]->digamma_vec;

                //double sum = 0; 
                //for (int k = 0; k < num_topics[t]; k++)
                //   sum += exp(dig[k]-digsum);

                //double frac = 1/((double)num_topics[t]);
                double sum = 1;
                double frac = 0;

                for (int k = 0; k < num_topics[t]; k++)
                    gsl_matrix_set(mat, dd, k, sum*(frac+exp(dig[k]-digsum))/(sum));
            }
            
            if(!isSmoothed)
            {
                // calling grad descent
               // gsl_vector *(gsl_matrix *D)
                //v = dirichlet_mle_descent(mat);
            }

            else
            **/
                //v = dirichlet_mle_s(mat, w, smoothWeight);

            // fill in values
            //double sum  = 0;
            /**
            for (int k = 0; k < num_topics[t]; k++)
            {
                as[t][a]->alpha_t[k] = gsl_vector_get (v, k);
                sum += as[t][a]->alpha_t[k];
                cout << "alpha k " <<  as[t][a]->alpha_t[k] << "\n";

            }
            as[t][a]->alpha_sum_t = sum;
            **/
        
    }
}

void slda::updatePriorSimple(double *** var_gamma, const settings * s)
{
    double isSmoothed = s->IS_SMOOTHED;
    double smoothWeight = s->SMOOTH_WEIGHT;

    for (int t = 0; t < num_word_types; t++)
    {
        cout << "T: " << t << "\n";
        double * p_doc = new double[num_topics[t]];
        double p_doc_sum = 0; 
        double * p_aut = new double[num_topics[t]];
        double p_aut_sum = 0; 

        for (int k = 0; k < num_topics[t]; k++)
        {
            p_doc[k] = 2*epsilon;
            p_doc_sum += 2*epsilon;
        }

        for (int d =0; d < num_docs; d++)
        {
            for (int k = 0; k < num_topics[t]; k++)
            {
                p_doc[k] += (var_gamma[d][t][k]-2*epsilon);
                p_doc_sum += (var_gamma[d][t][k]-2*epsilon);
            }
        }

        for (int k = 0; k < num_topics[t]; k++)
        {
            p_doc[k] /= p_doc_sum;
        }

        cout << "GLOBAL ALPHA: \n";
        for (int k = 0; k < num_topics[t]; k++)
        {
            cout << "alpha k " << k << ": " << p_doc[k] << "\n";
            as_global[t]->alpha_t[k] = p_doc[k];
        }

        as_global[t]->alpha_sum_t=p_doc_sum;


        //update

        cout << "AUTHOR ALPHA: \n";
        for (int a = 0; a < num_classes; a++)
        {
            cout << "AUTHOR ALPHA: " << a << " \n";

            p_aut_sum = 0;
            for (int k = 0; k < num_topics[t]; k++)
            {
                p_aut[k] = 2*epsilon;
                p_aut_sum += 2*epsilon;
            }
            for (int dd = 0; dd < docs_per[a]; dd++)
            {
                int d = getDoc(a,dd);
                for (int k = 0; k < num_topics[t]; k++)
                {
                    p_aut[k] += (var_gamma[d][t][k]-2*epsilon);
                    p_aut_sum += (var_gamma[d][t][k]-2*epsilon);
                }
            }

            for (int k = 0; k < num_topics[t]; k++)
            {
                p_aut[k] /= p_aut_sum;
            }

            for (int k = 0; k < num_topics[t]; k++)
            {
                cout << "alpha k " << k << ": " << p_aut[k] << "\n";
                as[t][a]->alpha_t[k] = p_aut[k];
            }
            as[t][a]->alpha_sum_t = p_aut_sum;

        }

    }
     // used for smoothing
}

void slda::globalPrior(double *** var_gamma, const settings *s)
{
    double isSmoothed = s->IS_SMOOTHED;
    double smoothWeight = s->SMOOTH_WEIGHT;

    for (int t = 0; t < num_word_types; t++)
    {
        if (s->ONE_TOPIC && t > 0)
        {
            as_global[t]->alpha_sum_t = as_global[t]->alpha_sum_t;
            for (int k = 0; k < num_topics[t]; k++)
            {
                    as_global[t]->alpha_t[k] = as_global[0]->alpha_t[k];
            }            
         }

        // used for smoothing
        gsl_vector * w;
        if (isSmoothed)
        {
            w = gsl_vector_alloc(num_topics[t]);
            for (int k = 0; k < num_topics[t]; k++)
                gsl_vector_set(w, k, 1/((double)(num_topics[t])));
        }
        gsl_matrix * mat = gsl_matrix_alloc(num_docs,num_topics[t]);
            gsl_vector * v;

        for (int d = 0; d < num_docs; d++ )
        {
            // digamma of sum
            double digsum = dg[t][d]->digamma_sum;
            std::vector<double> dig = dg[t][d]->digamma_vec;

            double sum = 0; 
            for (int k = 0; k < num_topics[t]; k++)
                sum += exp(dig[k]-digsum);

            double frac = 1/((double)num_topics[t]);


            for (int k = 0; k < num_topics[t]; k++)
                gsl_matrix_set(mat, d, k, sum*(frac+exp(dig[k]-digsum))/(1+sum));
        }
        if(!isSmoothed)
        {
            v = dirichlet_mle_descent(mat);
        }

        else
            v = dirichlet_mle_s(mat, w, smoothWeight);

        // fill in values
        double sum  = 0;
        for (int k = 0; k < num_topics[t]; k++)
        {
            as_global[t]->alpha_t[k] = gsl_vector_get (v, k);
            sum += as_global[t]->alpha_t[k];
        }
        as_global[t]->alpha_sum_t = sum;
        
    }
}

void slda::v_em(corpus * c, const settings * setting,
                const char * start, const char * directory)
{
    char filename[100];
    int * max_length = c->max_corpus_length();

    double likelihood, likelihood_old = 0, converged = 1;
    int d, n, i, t; // iterates
    double L2penalty = setting->PENALTY;

    // seed a random number generator for inference
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
    time_t seed;
    time(&seed);
    gsl_rng_set(rng, (long) seed);

    printf("initializing ...\n");

    std::vector<suffstats * > ss;
    for (t = 0; t < num_word_types; t++)
    {
        ss.push_back(new_suffstats(t));
        random_initialize_ss(ss[t], c,t );

        //other forms of initialization supported in original code
    }

    double rho = 1;
    mle_global(ss, -1, setting);
    std::vector<int>random_v;

    mle_logistic(ss, -1, setting, random_v);

    // allocate variational parameters
    double *** var_gamma, *** phi, ** lambda;
    var_gamma = new double ** [c->num_docs];
    for (d = 0; d < c->num_docs; d++)
    {
        var_gamma[d] = new double * [num_word_types];
    }

    phi = new double ** [num_word_types];
    for (t = 0; t < num_word_types; t++)
    {
        phi[t] = new double * [max_length[t]];
        for (d = 0; d < c->num_docs; d++)
        {
            var_gamma[d][t] = new double [num_topics[t]];
            for (int k = 0; k < num_topics[t]; k++)
                var_gamma[d][t][k] = .5 + ((double) (num_topics[t]))/
            ((double)c->docs[d]->total[t]);
        }

        for (n = 0; n < max_length[t]; n++)
            phi[t][n] = new double [num_topics[t]];
    }


    FILE * likelihood_file = NULL;
    sprintf(filename, "%s/likelihood.dat", directory);
    likelihood_file = fopen(filename, "w");

    int ETA_UPDATE = 0;

    i = 0;
    first_run = -1;
    double scaling = (double)(c->num_docs)/((double)setting->BATCH_SIZE);

    double converged_old = 99999;
    bool not_stop = true;
    while (not_stop || i<= 20)
    {

        printf("**** em iteration %d ****\n", ++i);
        for (t = 0; t < num_word_types; t++)
        {
            zero_initialize_ss(ss[t],t);
        }
        if (i > LDA_INIT_MAX)
            ETA_UPDATE = 1;
        // e-step
        printf("**** e-step ****\n");
        if ( setting->STOCHASTIC && i % 5 != 0 && i!=1)
        {
            //return;
            printf("STOCHASTIC\n");
            likelihood = 0;
            for (size_t iii = 0; iii < setting->BIG_BATCH_SIZE; iii ++)
            {
                //printf("BIG LOOP\n");
                for (t = 0; t < num_word_types; t++)
                {
                    zero_initialize_ss(ss[t],t);
                }
                std::vector<int> v;
                //sample docs
                for (size_t id = 0; id <setting->BATCH_SIZE;id++)
                {
                    v.push_back(gsl_rng_uniform_int(rng, c->num_docs));
                }
                for (size_t id = 0; id < setting->BATCH_SIZE; id++)
                {
                    d = v[id];
                    //if ((d % 100) == 0)
                    //    printf("document %d\n", d);
                    likelihood += slda_inference(c->docs[d], var_gamma[d], phi, as,  d, setting);
                    for(t=0; t< num_word_types; t++)
                    {
                        likelihood += doc_e_step(c->docs[d], var_gamma[d][t], phi[t],
                        ss[t], ETA_UPDATE, d,  t, 1, setting);
                    }
                }
               
                rho = (1/pow(1+10*i+iii,0.6));
                 //update dictionary and logistic parameter
                mle_global(ss, rho, setting);
                //mle_logistic(ss, rho, setting, v);
            }
            double scalar = ((double) c->num_docs)/((double)setting->BIG_BATCH_SIZE);
            likelihood = likelihood/((double)setting->BATCH_SIZE)*scalar;
        }
        else
        {
            likelihood = 0;
            for (d = 0; d < c->num_docs; d++)
            {
                if ((d % 100) == 0)
                    printf("document %d\n", d);
                likelihood += slda_inference(c->docs[d], var_gamma[d], phi, as,  d, setting);
                for (t=0; t< num_word_types; t++)
                {
                    likelihood += doc_e_step(c->docs[d], var_gamma[d][t], phi[t],
                     ss[t], ETA_UPDATE, d,  t, 1, setting);
                }
            }
            likelihood-=add_penalty(setting);
            mle_global(ss, -1, setting);
        }

        printf("likelihood: %10.10f\n", likelihood);
        // m-st
        printf("**** m-step ****\n");

        //update dictionary and logistic parameter
        if (i % 2 == 0)
        {
            cout << "MLE LOG \n";
            mle_logistic(ss, -1, setting, random_v);
        }

        /**
        if (setting->STOCHASTIC)
        {
            if (i % 3 == 0 && setting->ESTIMATE_ALPHA)
            {
                cout << "calling update prior\n";
                updatePrior(var_gamma, setting);
            }

        }
        **/
        if (i % 5 == 0 )//&& setting->ESTIMATE_ALPHA)
        {
            cout << "calling update prior\n";
            updatePrior(var_gamma, setting, c);
        }
        
        converged_old = converged;
        // check for convergence
        converged = ((likelihood_old - likelihood) / (likelihood_old));
        //if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2;
        likelihood_old = likelihood;

        //bool stop = (converged < 0);
        not_stop = (converged > setting->EM_CONVERGED);
        not_stop = not_stop || (converged_old > setting->EM_CONVERGED);
        not_stop = not_stop || (i <= setting->EM_MIN_ITER);
        cout << "not_stop" << not_stop << " \n";

        not_stop = not_stop && (i <= setting->EM_MAX_ITER);
        // output model and likelihood
        fprintf(likelihood_file, "%10.10f\t%5.5e\n", likelihood, converged);
        fflush(likelihood_file);
        if ((i % LAG) == 0)
        {
            sprintf(filename, "%s/%03d.model", directory, i);
            save_model(filename);
            sprintf(filename, "%s/%03d.gamma", directory, i);
            save_gamma(filename, var_gamma, c->num_docs);

            sprintf(filename, "%s/%03d.model.text", directory, i);
            save_model_text(filename);
        }
    }

    if (setting->ESTIMATE_ALPHA)
    {
        updatePriorSimple(var_gamma, setting);
        //globalPrior(var_gamma, setting);
    }
    // output the final model
    sprintf(filename, "%s/final.model", directory);
    save_model(filename);
    sprintf(filename, "%s/final.gamma", directory);
    //save_gamma(filename, var_gamma, c->num_docs);

    sprintf(filename, "%s/final.model.text", directory);
    save_model_text(filename);


    fclose(likelihood_file);
    FILE * w_asgn_file = NULL;
    sprintf(filename, "%s/word-assignments.dat", directory);
    w_asgn_file = fopen(filename, "w");
    
    for (d = 0; d < c->num_docs; d++)
    {
        //final inference
        if ((d % 100) == 0)
            printf("final e step document %d\n", d);

        likelihood += slda_inference(c->docs[d], var_gamma[d], phi, as, d,setting);
        write_word_assignment(w_asgn_file, c->docs[d], phi);

    }

    likelihood -= add_penalty(setting);

    //printf("FINAL LOG perplexity %f", perplexity);

    fclose(w_asgn_file);

    for (d = 0; d < c->num_docs; d++)
    {
        for (t = 0; t < num_word_types; t++)
            delete [] var_gamma[d][t];
        delete [] var_gamma[d];
    }
    delete [] var_gamma;

    //~std::vector<suffstats *>(ss);
    for (t = 0; t < num_word_types; t++)
    {
        for (n = 0; n < max_length[t]; n++)
            delete [] phi[t][n];
        delete [] phi[t];
    }
    delete [] phi;
}

/*
 * Maps all values of eta into indices for one long vector
 * t: word type
 * l: class number
 * k: word number inside class
 */
int slda::vec_index(int t, int l, int k)
{
    int output = 0;
    for (size_t ti = 0; ti < t; ti++)
    {
        output += (num_classes - 1) * (num_topics[ti]);
    }

    output += l * num_topics[t];

    output += k;

    return output;
}

void slda::mle_global(vector<suffstats *> ss, double rho, const settings * setting)
{
    int k, w, t;

    //printf("rho! %f",rho);

    for (t = 0; t < num_word_types; t++)
        for (k = 0; k < num_topics[t]; k++)
        {
            if (setting->TOPIC_SMOOTH)
            {
                //cout << "SMOOTH TOPICS \n";
                //not stochastic
                if (!setting->STOCHASTIC || first_run == 1)
                    for (w = 0; w < size_vocab[t]; w++)
                    {
                        lambda[t][k][w] = top_prior + ss[t]->word_ss[k][w];
                    }
                //stochastic
                else
                {
                    double lambda_old, lambda_new;
                    for (w = 0; w < size_vocab[t]; w++)
                    {
                        lambda_new = top_prior + ss[t]->word_ss[k][w];
                        lambda_old = lambda[t][k][w];
                        lambda[t][k][w] = lambda_new*rho + (1-rho)*lambda_old;
                    }
                }

                double psi_sum = 0;
                // fill in log_prob_w equivalence
                for (w = 0; w < size_vocab[t]; w++)
                {

                    log_prob_w[t][k][w] = digamma(lambda[t][k][w]);
                    psi_sum += lambda[t][k][w];
                }
                psi_sum = digamma(psi_sum);
                for (w = 0; w < size_vocab[t]; w++)
                {
                    log_prob_w[t][k][w] -= psi_sum;
                }
            }
            //no smoothing
            else
            {
                for (w = 0; w < size_vocab[t]; w++)
                {
                    if (ss[t]->word_ss[k][w] > 0) 
                    {
                        log_prob_w[t][k][w] = (double)log(ss[t]->word_ss[k][w]) - log(ss[t]->word_total_ss[k]);
                    } 
                    else 
                    {
                        log_prob_w[t][k][w] = -100.0;
                    }
                }
            }
        }   
}


void slda::mle_logistic(std::vector<suffstats *> ss, double rho, 
    const settings * setting, std::vector<int> docs)
{
    int k, w, t;
    //the label part goes here
    double f = 0.0;
    int status;
    int opt_iter;

    // the total length of all the eta parameters
    int opt_size = 0;
    for (t = 0; t< num_word_types; t++)
        opt_size += num_topics[t];
    opt_size = opt_size * (num_classes-1);

    int l;

    opt_parameter param;
    param.ss = ss;
    param.model = this;
    param.PENALTY = setting->PENALTY;
    param.L1_PENALTY =  setting->L1_PENALTY;
    gsl_vector * x;


    if (setting->STOCHASTIC && rho > 0)
    {
        x = gsl_vector_alloc(opt_size);
        // allocate the long vector of eta values to be optimized
        for(t = 0; t < num_word_types; t++)
        {  
            for (l = 0; l < num_classes-1; l++)
                for (k = 0; k < num_topics[t]; k++)
                    gsl_vector_set(x, vec_index(t,l,k), eta[t][l][k]);
        }
        //initialize the stochastic elements 
        gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
        time_t seed;
        time(&seed);
        gsl_rng_set(rng, (long) seed);
        stoch_opt_parameter par2;

        if (first_run == 1)
            rho = 1;

        par2.op = &param;
        par2.stoch_docs = docs;
        par2.doc_trials = docs.size();
        // loop over batches?
        gsl_vector * df = gsl_vector_alloc(opt_size);
        softmax_df_stoch(x, (void *) &par2, df);

        for (size_t tt = 0; tt < 10; tt++)
        {
            for (t = 0; t < num_word_types; t++)
                for (l = 0; l < num_classes-1; l++)
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        eta[t][l][k] -= rho*gsl_vector_get(df,vec_index(t,l,k));
                        if(setting->USE_L1)
                            eta[t][l][k] -= sign_v(eta[t][l][k])/(1+tt);
                    }
            for (l = 0; l < num_classes-1; l++)
                for (k = 0; k < num_topics[t]; k++)
                    gsl_vector_set(x, vec_index(t,l,k), eta[t][l][k]);
        }
        //double f = softmax_f(x,(void *) &param);
        //printf("final f: %f\n", f);

        return;

    }
    printf("Maximizing ...\n");

    const gsl_multimin_fdfminimizer_type * T;
    gsl_multimin_fdfminimizer * s;
    gsl_multimin_function_fdf opt_fun;
    opt_fun.f = &softmax_f;
    opt_fun.df = &softmax_df;
    opt_fun.fdf = &softmax_fdf;
    opt_fun.n = opt_size;
    opt_fun.params = (void*)(&param);
    x = gsl_vector_alloc(opt_size);

// allocate the long vector of eta values to be optimized
    for (t = 0; t < num_word_types; t++)
    {  
        for (l = 0; l < num_classes-1; l++)
            for (k = 0; k < num_topics[t]; k++)
                gsl_vector_set(x, vec_index(t,l,k), eta[t][l][k]);
    }


    /// L1 Logistic Regression
    //if (setting->USE_L1)
    //{
        //lgbfs_opt(param, x);
    //}

    // normal optimization, possiby with ridge

    T = gsl_multimin_fdfminimizer_vector_bfgs;
    s = gsl_multimin_fdfminimizer_alloc(T, opt_size);
    gsl_multimin_fdfminimizer_set(s, &opt_fun, x, 0.02, 1e-4);

    opt_iter = 0;
    do
    {
        opt_iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);
        if (status)
            break;
        status = gsl_multimin_test_gradient(s->gradient, 1e-3);
        if (status == GSL_SUCCESS)
            break;
        f = -s->f;
        if ((opt_iter-1) % 10 == 0)
            printf("step: %02d -> f: %f\n", opt_iter-1, f);
    } while (status == GSL_CONTINUE && opt_iter < MSTEP_MAX_ITER);


    for (t = 0; t < num_word_types; t++)
    {
        for (l = 0; l < num_classes-1; l++)
            for (k = 0; k < num_topics[t]; k++)
                eta[t][l][k] = gsl_vector_get(s->x, vec_index(t,l,k));
    }

    gsl_multimin_fdfminimizer_free (s);
    gsl_vector_free (x);

    printf("final f: %f\n", f);
}


double slda::add_penalty(const settings * setting)
{
    double PENALTY = setting->PENALTY;
    double L1_PENALTY = setting->L1_PENALTY;
    int t, l,k;
    double output = 0;
    for (t = 0; t < num_word_types; t++)
        for (l = 0; l < num_classes-1; l++)
            for (k = 0; k < num_topics[t]; k++)
            {
                output += pow(eta[t][l][k], 2) * PENALTY/2.0;
                output += abs_v(eta[t][l][k]) * L1_PENALTY;
            }
    return output;

}

double slda::doc_e_step(document* doc, double* gamma, double** phi,
                        suffstats * ss, int eta_update, int _docNum, int t,
                        double scaling, const settings * setting)
{
    double likelihood = 0.0;
    //if (eta_update == 1)
    //    likelihood = slda_inference(doc, gamma, phi, setting);
    //else
    //   likelihood = lda_inference(doc, gamma, phi, setting);

    int d = _docNum;

    int n, k, i, idx, m;

    int a = doc->label;

    // update sufficient statistics

    int num_var_entries = num_topics[t]*(num_topics[t]+1)/2;
    if (setting->STOCHASTIC)
    {
        memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics[t]);
        memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);
    }

    for (n = 0; n < doc->length[t]; n++)
    {
        for (k = 0; k < num_topics[t]; k++)
        {
            // scaling used for stochastic variational inference
            ss->word_ss[k][doc->words[t][n]] += scaling*doc->counts[t][n]*phi[n][k];
            ss->word_total_ss[k] += scaling*doc->counts[t][n]*phi[n][k];

            //statistics for each document of the supervised part
            ss->z_bar[d].z_bar_m[k] += doc->counts[t][n] * phi[n][k]; //mean
            for (i = k; i < num_topics[t]; i++)  //variance
            {
                idx = map_idx(k, i, num_topics[t]);
                if (i == k)
                    ss->z_bar[d].z_bar_var[idx] +=
                        doc->counts[t][n] * doc->counts[t][n] * phi[n][k];

                ss->z_bar[d].z_bar_var[idx] -=
                    doc->counts[t][n] * doc->counts[t][n] * phi[n][k] * phi[n][i];
            }
        }
    }
    for (k = 0; k < num_topics[t]; k++)
    {
        ss->z_bar[d].z_bar_m[k] /= (double)(doc->total[t]);
    }
    for (i = 0; i < num_topics[t]*(num_topics[t]+1)/2; i++)
    {
        ss->z_bar[d].z_bar_var[i] /= (double)(doc->total[t] * doc->total[t]);
    }

    //ss->num_docs = ss->num_docs + 1; //because we need it for store statistics for each docs

    return (likelihood);
}

double slda::doc_perplexity(document* doc, double ** expAlpha, double *** phi)
{
    int k, n, l, t;
    double perplexity = 0;

    for (t = 0; t < num_word_types; t++)
    {
        for (n = 0; n < doc->length[t]; n++)
        {
            double temp = 0;
            temp= log_prob_w[t][0][doc->words[t][n]] + phi[t][n][0];
            for (k = 1; k < num_topics[t]; k++)      
            {
                temp = log_sum(temp, log_prob_w[t][0][doc->words[t][n]] + phi[t][n][0]); 
            }
            perplexity+=temp;
        }
        
        for (k = 0; k < num_topics[t]; k++)
            perplexity+= doc->length[t]*expAlpha[t][k];
    }
    return perplexity;
}



/**
double slda::auth_perplexity(document* doc, double ** var gamma, )
{
    int k, n, l, t;
}
**/
double slda::slda_compute_likelihood(document* doc, double*** phi,
    double** var_gamma, int d)
{
    double likelihood = 0, var_gamma_sum = 0, t0 = 0.0, t1 = 0.0, t2 = 0.0;

    int k, n, l, t;
    double ** dig = new  double * [num_word_types];
    for (t=0; t< num_word_types; t++)
        dig[t] = new double [num_topics[t]];

    double * digsum = new double [num_word_types];
    for (t=0; t< num_word_types; t++)
        digsum[t] = 0;

    int flag;

    int a = doc->label;

    for (t = 0; t < num_word_types; t++)
    {
        // computes digamma terms for the gamma variation parameters
        var_gamma_sum = 0;
        for (k = 0; k < num_topics[t]; k++)
        {
            dig[t][k] = digamma(var_gamma[t][k]);
            dg[t][d]->digamma_vec[k] = dig[t][k];

            var_gamma_sum += var_gamma[t][k];
        }
        digsum[t] = digamma(var_gamma_sum);
        dg[t][d]->digamma_sum = digsum[t];
        //computes contribution of Dirichlet paraemter sums to log likelihood
        likelihood += lgamma((as[t][a]->alpha_sum_t)) - lgamma(var_gamma_sum);
    }

    //t0 stors likelihood contributions
    for (t= 0; t < num_word_types; t++)
    {
        t0 = 0.0;
        for (k = 0; k < num_topics[t]; k++)
        {
            likelihood += -lgamma(as[t][a]->alpha_t[k])
                          + (as[t][a]->alpha_t[k] - 1)*(dig[t][k] - digsum[t]) +
                          lgamma(var_gamma[t][k]) - (var_gamma[t][k] - 1)*(dig[t][k] - digsum[t]);

            for (n = 0; n < doc->length[t]; n++)
            {
                if (phi[t][n][k] > 0)
                {
                    likelihood += doc->counts[t][n]*(phi[t][n][k]*((dig[t][k] - digsum[t]) - log(phi[t][n][k]) + log_prob_w[t][k][doc->words[t][n]]));
                    if (doc->label < num_classes-1)
                        t0 += eta[t][doc->label][k] * doc->counts[t][n] * phi[t][n][k];
                }
            }
        }
        likelihood += t0 / (double)(doc->total[t]);     //eta_k*\bar{\phi}
    }

    t0 = 1.0; //the class model->num_classes-1
    for (l = 0; l < num_classes-1; l++)
    {
        t1 = 1.0;
        for (t = 0; t < num_word_types; t++)
        {
            for (n = 0; n < doc->length[t]; n++)
            {
                t2 = 0.0;
                for (k = 0; k < num_topics[t]; k++)
                {
                    t2 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                }
                t1 *= t2;
            }
        }
        t0 += t1;
    }
    likelihood -= log(t0);
    for (t=0; t< num_word_types; t++)
        delete[] dig[t];
    delete [] dig;
    delete [] digsum;
    //printf("%lf\n", likelihood);
    return likelihood;
}


double slda::slda_inference_1(document* doc, double ** var_gamma, double *** phi,
                            alphas *** as, int d, const settings * setting)
{
    int k, n, var_iter, l,t;
    int FP_MAX_ITER = 10;
    int fp_iter = 0;

    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;
    double t0 = 0.0;

    // fix this
    double **digamma_gam = new double * [num_word_types];
    double *sf_aux = new double [num_classes-1];

    for (t = 0; t < num_word_types; t++)
        digamma_gam[t] = new double [num_topics[t]];

    double sf_val = 0.0;
    int a = doc->label;

    // compute posterior dirichlet

    int full_total = 0;
    for (t = 0; t < num_word_types; t++)
        full_total+= doc->total[t];

    // iterate over ole
    for (t = 0; t < 1; t++)
    {
        for (k = 0; k < num_topics[t]; k++)
        {
            var_gamma[t][k] = 0;
            var_gamma[t][k] = as[t][a]->alpha_t[k] + (full_total/((double) num_topics[t]));
            digamma_gam[t][k] = digamma(var_gamma[t][k]);
            for (n = 0; n < doc->length[t]; n++)
                phi[t][n][k] = 1.0/(double)(num_topics[t]);

        }
    }


    for (t = 1; t < num_word_types; t++)
    {
        for (k = 0; k < num_topics[t]; k++)
        {
            var_gamma[t][k] = var_gamma[0][k];
            digamma_gam[t][k] = digamma(var_gamma[0][k]);
            for (n = 0; n < doc->length[t]; n++)
                phi[t][n][k] = 1.0/(double)(num_topics[t]);

        }
    }

    for (l = 0; l < num_classes-1; l++)
    {
        sf_aux[l] = 1.0; // the quantity for equation 6 of each class

        for (t = 0; t < num_word_types; t++)
            for (n = 0; n < doc->length[t]; n++)
            {
                t0 = 0.0;
                for (k = 0; k < num_topics[t]; k++)
                {
                    t0 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                }
                sf_aux[l] *= t0;
            }
    }

    var_iter = 0;

    while ((converged > setting->VAR_CONVERGED) && ((var_iter < setting->VAR_MAX_ITER) || (setting->VAR_MAX_ITER == -1)))
    {
        var_iter++;

        for (t = 0; t < num_word_types; t++)
        {
            double * sf_params = new double [num_topics[t]];
            double * oldphi = new double [num_topics[t]];

            for (n = 0; n < doc->length[t]; n++)
            {
                //compute sf_params
                memset(sf_params, 0, sizeof(double)*num_topics[t]);


                //in log space
                for (l = 0; l < num_classes-1; l++)
                {
                    t0 = 0.0;
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        t0 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                    }
                    sf_aux[l] /= t0; //take out word n

                    for (k = 0; k < num_topics[t]; k++)
                    {
                        //h in the paper
                        sf_params[k] += sf_aux[l]*exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                    }
                }

                for (k = 0; k < num_topics[t]; k++)
                {
                    oldphi[k] = phi[t][n][k];
                }

                for (fp_iter = 0; fp_iter < FP_MAX_ITER; fp_iter++)  //fixed point update
                {
                    sf_val = 1.0; // the base class, in log space
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        sf_val += sf_params[k]*phi[t][n][k];
                    }

                    phisum = 0;
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        phi[t][n][k] = digamma_gam[0][k] + log_prob_w[t][k][doc->words[t][n]];

                        //added softmax parts
                        if (doc->label < num_classes-1)
                            phi[t][n][k] += eta[t][doc->label][k]/(double)(doc->total[t]);
                        phi[t][n][k] -= sf_params[k]/(sf_val*(double)(doc->counts[t][n]));

                        if (k > 0)
                            phisum = log_sum(phisum, phi[t][n][k]);
                        else
                            phisum = phi[t][n][k]; // note, phi is in log space
                    }
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        phi[t][n][k] = exp(phi[t][n][k] - phisum); //normalize
                    }
                }
                //back to sf_aux value
                for (l = 0; l < num_classes-1; l++)
                {
                    t0 = 0.0;
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        t0 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                    }
                    sf_aux[l] *= t0;
                }
                for (k = 0; k < num_topics[t]; k++)
                {
                    var_gamma[t][0] = var_gamma[t][0] + doc->counts[t][n]*(phi[t][n][k] - oldphi[k]);
                    digamma_gam[t][0] = digamma(var_gamma[t][0]);
                }
            }

        }

        for (t = 1; t < num_word_types; t++)
        {
            for (k = 0; k < num_topics[t]; k++)
            {
                var_gamma[t][k] = var_gamma[0][k];
                digamma_gam[t][k] = digamma(var_gamma[0][k]);
           
            }
        }

        likelihood = slda_compute_likelihood(doc, phi, var_gamma, d);
        assert(!isnan(likelihood));
        converged = fabs((likelihood_old - likelihood) / likelihood_old);
        likelihood_old = likelihood;

        // manage this memory

    }

    for (t = 0; t < num_word_types; t++)
        delete [] digamma_gam[t];
    delete [] digamma_gam;
    delete [] sf_aux;

    return likelihood;
}

double slda::slda_inference(document* doc, double ** var_gamma, double *** phi,
                            alphas *** as, int d, const settings * setting)
{

    if(setting->ONE_TOPIC)
    {
        return slda_inference_1( doc, var_gamma,  phi,
                            as,  d,  setting);
    }
    int k, n, var_iter, l,t;
    int FP_MAX_ITER = 10;
    int fp_iter = 0;

    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;
    double t0 = 0.0;

    // fix this
    double **digamma_gam = new double * [num_word_types];
    double *sf_aux = new double [num_classes-1];

    for (t = 0; t < num_word_types; t++)
        digamma_gam[t] = new double [num_topics[t]];

    double sf_val = 0.0;
    int a = doc->label;

    // compute posterior dirichlet
    for (t = 0; t < num_word_types; t++)
    {
        for (k = 0; k < num_topics[t]; k++)
        {
            if( var_gamma[t][k] == 0 || setting->ORIGINAL)
            {
                var_gamma[t][k] = as[t][a]->alpha_t[k] + (doc->total[t]/((double) num_topics[t]));
               // if (d == 0 && k == 0 && t == 0)
               // {
               //    cout << "zero shift \n";
                //}
            }
            digamma_gam[t][k] = digamma(var_gamma[t][k]);
            for (n = 0; n < doc->length[t]; n++)
                phi[t][n][k] = 1.0/(double)(num_topics[t]);
        }
    }

    for (l = 0; l < num_classes-1; l++)
    {
        sf_aux[l] = 1.0; // the quantity for equation 6 of each class

        for (t = 0; t < num_word_types; t++)
            for (n = 0; n < doc->length[t]; n++)
            {
                t0 = 0.0;
                for (k = 0; k < num_topics[t]; k++)
                {
                    t0 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                }
                sf_aux[l] *= t0;
            }
    }

    var_iter = 0;

    while ((converged > setting->VAR_CONVERGED) && ((var_iter < setting->VAR_MAX_ITER) || (setting->VAR_MAX_ITER == -1)))
    {
        var_iter++;

        for (t = 0; t < num_word_types; t++)
        {
            double * sf_params = new double [num_topics[t]];
            double * oldphi = new double [num_topics[t]];

            for (n = 0; n < doc->length[t]; n++)
            {
                //compute sf_params
                memset(sf_params, 0, sizeof(double)*num_topics[t]);


                //in log space
                for (l = 0; l < num_classes-1; l++)
                {
                    t0 = 0.0;
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        t0 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                    }
                    sf_aux[l] /= t0; //take out word n

                    for (k = 0; k < num_topics[t]; k++)
                    {
                        //h in the paper
                        sf_params[k] += sf_aux[l]*exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                    }
                }

                for (k = 0; k < num_topics[t]; k++)
                {
                    oldphi[k] = phi[t][n][k];
                }

                for (fp_iter = 0; fp_iter < FP_MAX_ITER; fp_iter++)  //fixed point update
                {
                    sf_val = 1.0; // the base class, in log space
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        sf_val += sf_params[k]*phi[t][n][k];
                    }

                    phisum = 0;
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        phi[t][n][k] = digamma_gam[t][k] + log_prob_w[t][k][doc->words[t][n]];

                        //added softmax parts
                        if (doc->label < num_classes-1)
                            phi[t][n][k] += eta[t][doc->label][k]/(double)(doc->total[t]);
                        phi[t][n][k] -= sf_params[k]/(sf_val*(double)(doc->counts[t][n]));

                        if (k > 0)
                            phisum = log_sum(phisum, phi[t][n][k]);
                        else
                            phisum = phi[t][n][k]; // note, phi is in log space
                    }
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        phi[t][n][k] = exp(phi[t][n][k] - phisum); //normalize
                    }
                }
                //back to sf_aux value
                for (l = 0; l < num_classes-1; l++)
                {
                    t0 = 0.0;
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        t0 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                    }
                    sf_aux[l] *= t0;
                }
                if (setting->ORIGINAL)
                    for (k = 0; k < num_topics[t]; k++)
                    {
                        var_gamma[t][k] = var_gamma[t][k] + doc->counts[t][n]*(phi[t][n][k] - oldphi[k]);
                        digamma_gam[t][k] = digamma(var_gamma[t][k]);
                    }
                
            }

        }
        if (!setting->ORIGINAL)
            for (t = 0; t < num_word_types; t++)
            {
                for (k = 0; k < num_topics[t]; k++)
                {
                    var_gamma[t][k] = as[t][a]->alpha_t[k];
                    for (n = 0; n < doc->length[t]; n++)
                    {
                         var_gamma[t][k]+= phi[t][n][k]*doc->counts[t][n];
                    }
                    digamma_gam[t][k] = digamma(var_gamma[t][k]);

                }
            }
        


        likelihood = slda_compute_likelihood(doc, phi, var_gamma, d);
        assert(!isnan(likelihood));
        converged = fabs((likelihood_old - likelihood) / likelihood_old);
        likelihood_old = likelihood;

        // manage this memory

    }

    for (t = 0; t < num_word_types; t++)
        delete [] digamma_gam[t];
    delete [] digamma_gam;
    delete [] sf_aux;

    return likelihood;
}

// descending order comparator
bool slda::mComp ( const mypair& l, const mypair& r)
        { return l.first > r.first; }

void slda::infer_only(corpus * c, const settings * setting, const char * directory)
{
    int i, k, d, n, t;
    double ** phi_m;
    double base_score, score;
    int label;
    int num_correct = 0;

    std::vector<int> recallAt;
    for (i = 0; i < 2; i++)
        recallAt.push_back(0);
    if (num_classes > 2)
         recallAt.push_back(0);

    char filename[100];
    int * max_length = c->max_corpus_length();
    double ***var_gamma, ***phi;
    double likelihood, likelihood_old = 0, converged = 1;

    // allocate variational parameters
    var_gamma = new double ** [c->num_docs];
    for (d = 0; d < c->num_docs; d++)
        var_gamma[d] = new double * [num_word_types];

    phi = new double ** [num_word_types];
    phi_m = new double * [num_word_types];

    for (t = 0; t < num_word_types; t++)
    {
        //std::cout << "t:" << t << "\n";
        //std::cout << "num_topics[t]:" << num_topics[t] << "\n";
        phi_m[t] = new double [num_topics[t]];

        phi[t] = new double * [max_length[t]];
        for (d = 0; d < c->num_docs; d++)
        {    
            var_gamma[d][t] = new double [num_topics[t]];
        }

        for (n = 0; n < max_length[t]; n++)
        {
            phi[t][n] = new double [num_topics[t]];
        }
    }

    FILE * likelihood_file = NULL;
    sprintf(filename, "%s/inf-likelihood.dat", directory);
    likelihood_file = fopen(filename, "w");
    FILE * inf_label_file = NULL;
    sprintf(filename, "%s/inf-labels.dat", directory);
    inf_label_file = fopen(filename, "w");

    double perplexity=0;
    int bigTotal=0;

    double *** expAlpha = new double ** [num_classes];

    for (i = 0; i < num_classes; i++)
    {
        expAlpha[i] =  new double *[num_word_types];
        for (t = 0; t < num_word_types; t++)
        {
            expAlpha[i][t] = new double [num_topics[t]];

            double psi_sum = digamma(as[t][i]->alpha_sum_t);
            for (k = 0; k < num_topics[t]; k++)
            {
                expAlpha[i][t][k] = digamma(as[t][i]->alpha_t[k]) - psi_sum;
            }
        }
    }


    for (d = 0; d < c->num_docs; d++)
    {
        if ((d % 100) == 0)
            printf("document %d\n", d);

        document * doc = c->docs[d];
        likelihood = 0;

        for (t = 0; t < num_word_types; t++)
        {

            likelihood += lda_inference(doc, var_gamma[d][t], phi[t], setting,t,-1);

            memset(phi_m[t], 0, sizeof(double)*num_topics[t]); //zero_initialize
            for (n = 0; n < doc->length[t]; n++)
            {
                for (k = 0; k < num_topics[t]; k++)
                {
                    phi_m[t][k] += doc->counts[t][n] * phi[t][n][k];
                }
            }
            for (k = 0; k < num_topics[t]; k++)
            {
                phi_m[t][k] /= (double)(doc->total[t]);
            }
        }

        perplexity += doc_perplexity(doc, expAlpha[doc->label], phi);
        // sum of total words
        for (t = 0; t < num_word_types; t++)
            bigTotal+= doc->total[t];

        //do classification

    
        std::vector<mypair> labels;
        label = num_classes-1;
        base_score = 0.0;
        labels.push_back(mypair(0.0,num_classes-1));
        for (i = 0; i < num_classes-1; i++)
        {
            score = 0.0;

            for (t = 0; t < num_word_types; t++)
                for (k = 0; k < num_topics[t]; k++)
                    score += eta[t][i][k] * phi_m[t][k];

            labels.push_back(mypair(score,i));

            if (score > base_score)
            {
                base_score = score;
                label = i;
            }
        }

        if (label == doc->label)
            num_correct++;

        //sort labels by score
        std::sort(labels.begin(), labels.end(), mComp);
        for (size_t i1 = 0; i1 < recallAt.size(); i1++)
        {
            // if a label is correct, then recall at i2>=i1 is incremented
            if (labels[i1].second == doc->label)
            {
                for (size_t i2 = i1; i2 < recallAt.size(); i2++)
                    recallAt[i2]++;
                break;
            }
        }

        fprintf(likelihood_file, "%5.5f\n", likelihood);
        //printf( "%5.5f\n", likelihood);
        fprintf(inf_label_file, "%d\n", label);
    }

    printf("average accuracy: %.3f\n", (double)num_correct / (double) c->num_docs);
    fprintf(inf_label_file,"average accuracy: %.3f\n", (double)num_correct / (double) c->num_docs);

    for (i = 0; i < recallAt.size(); i++)
    {
        fprintf(inf_label_file,"recall at %d: %.3f\n", i+1, (double)recallAt[i] / (double) c->num_docs);
        printf("recall at %d: %.3f\n", i+1, (double)recallAt[i] / (double) c->num_docs);
    }

    sprintf(filename, "%s/inf-gamma.dat", directory);
    save_gamma(filename, var_gamma, c->num_docs);
    perplexity = -1*perplexity/((double) bigTotal);
    printf("log perplexity %f", perplexity);
    fprintf(inf_label_file,"log perplexity %f\n", perplexity);


    for (d = 0; d < c->num_docs; d++)
    {
        for (t = 0; t < num_word_types; t++)
            delete [] var_gamma[d][t];
        delete [] var_gamma[d];
    }
    delete [] var_gamma;

    for (t = 0; t < num_word_types; t++)
    {
        for (n = 0; n < max_length[t]; n++)
            delete [] phi[t][n];
        delete [] phi[t];
    }
    delete [] phi;

    for (t = 0; t < num_word_types; t++)
        delete [] phi_m[t];
    delete [] phi_m[t];
}

// maximum posterior likelihood approach to finding authorship
void slda::infer_only_2(corpus * c, const settings * setting, const char * directory)
{
    int i, k, d, n, t;
    double ** phi_m;
    double base_score, score;
    int label, var_iter;
    int num_correct = 0;

    init_dg(c->num_docs);

    std::vector<int> recallAt;
    for (i = 0; i < 2; i++)
        recallAt.push_back(0);
    if (num_classes > 2)
        recallAt.push_back(0);

    char filename[100];
    int * max_length = c->max_corpus_length();
    double ***var_gamma, ***phi;
    double likelihood, likelihood_old = 0, converged = 1;

    // allocate variational parameters
    var_gamma = new double ** [c->num_docs];
    for (d = 0; d < c-> num_docs; d++)
        var_gamma[d] = new double * [num_word_types];

    phi = new double ** [num_word_types];
    phi_m = new double * [num_word_types];

    for (t = 0; t < num_word_types; t++)
    {
        phi_m[t] = new double [num_topics[t]];

        phi[t] = new double * [max_length[t]];
        for (d = 0; d < c->num_docs; d++)
        {
            var_gamma[d][t] = new double [num_topics[t]];
            for (k = 0; k < num_topics[t]; k++)
                var_gamma[d][t][k] = 0;
        }

        for (n = 0; n < max_length[t]; n++)
        {
            phi[t][n] = new double [num_topics[t]];
            for (k = 0; k < num_topics[t]; k++)
                phi[t][n][k] = 0;
        }
    }

    FILE * likelihood_file = NULL;
    sprintf(filename, "%s/inf-likelihood.dat", directory);
    likelihood_file = fopen(filename, "w");
    FILE * inf_label_file = NULL;
    sprintf(filename, "%s/inf-labels.dat", directory);
    inf_label_file = fopen(filename, "w");

    gsl_vector * lkhoods = gsl_vector_alloc(num_classes);
    int true_label = -1;
    converged = 99;


    for (d = 0; d < c->num_docs; d++)
    {
        //if ((d % 100) == 0)
         //   printf("document %d\n", d);

        // store documents actual label
        document * doc = c->docs[d];
        true_label = doc->label;

        // see which author maximizes the posterior likelihood
        for (int a = 0; a < num_classes; a++)
        {

            // pretend doc is written by author a
            doc->label = a;
            likelihood_old = 1;
            converged = 99;
            var_iter = 0;
            // fit until convergence
            while (converged > setting->VAR_CONVERGED
                   && (var_iter < setting->VAR_MAX_ITER || setting->VAR_MAX_ITER == -1))
            {
                // reset likelihood
                likelihood = slda_inference(c->docs[d], var_gamma[d], phi, as,  d, setting);

                converged = fabs((likelihood_old - likelihood) / (likelihood_old));
                likelihood_old = likelihood;
                var_iter += 1;
            }
            gsl_vector_set(lkhoods, a, likelihood);

        }

        doc->label = true_label;     //reset original document label

        int label = gsl_vector_max_index(lkhoods);
        std::vector<mypair> labels;
        label = 0;
        base_score  = lkhoods->data[0];

        for (i = 0; i < num_classes-1; i++)
        {
            score =lkhoods->data[i];

            labels.push_back(mypair(score,i));

            if (score > base_score)
            {
                base_score = score;
                label = i;
            }
        }

        if (label == doc->label)
            num_correct++;

        //sort labels by score
        std::sort(labels.begin(), labels.end(), mComp);
        for (size_t i1 = 0; i1 < recallAt.size(); i1++)
        {
            // if a label is correct, then recall at i2>=i1 is incremented
            if (labels[i1].second == doc->label)
            {
                for (size_t i2 = i1; i2 < recallAt.size(); i2++)
                    recallAt[i2]++;
                break;
            }
        }

        fprintf(likelihood_file, "likelihood: %5.5f\n", likelihood);

        fprintf(inf_label_file, "%d\n", label);
    }


    printf("average accuracy: %.3f\n", (double)num_correct / (double) c->num_docs);
    fprintf(inf_label_file,"average accuracy: %.3f\n", (double)num_correct / (double) c->num_docs);

    for (i = 0; i < recallAt.size(); i++)
    {
        fprintf(inf_label_file,"recall at %d: %.3f\n", i+1, (double)recallAt[i] / (double) c->num_docs);
        printf("recall at %d: %.3f\n", i+1, (double)recallAt[i] / (double) c->num_docs);
    }

    sprintf(filename, "%s/inf-gamma.dat", directory);
    save_gamma(filename, var_gamma, c->num_docs);
    //perplexity = -1*perplexity/((double) bigTotal);
    //printf("log perplexity %f", perplexity);
    //fprintf(inf_label_file,"log perplexity %f\n", perplexity);

    for (d = 0; d < c->num_docs; d++)
    {
        for (t = 0; t < num_word_types; t++)
            delete [] var_gamma[d][t];
        delete [] var_gamma[d];
    }
    delete [] var_gamma;

    for (t = 0; t < num_word_types; t++)
    {
        for (n = 0; n < max_length[t]; n++)
            delete [] phi[t][n];
        delete [] phi[t];
    }
    delete [] phi;

    for (t = 0; t < num_word_types; t++)
        delete [] phi_m[t];
    delete [] phi_m[t];
}


void slda::save_gamma(char* filename, double*** gamma, int num_docs)
{
    int d, k, t;

    FILE* fileptr = fopen(filename, "w");
    for (t = 0; t < num_word_types; t++)
        for (d = 0; d < num_docs; d++)
        {
            fprintf(fileptr, "%5.10f", gamma[d][t][0]);
            for (k = 1; k < num_topics[t]; k++)
                fprintf(fileptr, " %5.10f", gamma[d][t][k]);
            fprintf(fileptr, "\n");
        }
    fprintf(fileptr, "\n");
    fclose(fileptr);
}


void slda::write_word_assignment(FILE* f, document* doc, double*** phi)

{
    int n, t;

    for (t = 0; t < num_word_types; t++)
    {
        fprintf(f, "%03d", doc->length[t]);
        for (n = 0; n < doc->length[t]; n++)
        {
            fprintf(f, " %04d:%02d", doc->words[t][n], argmax(phi[t][n], num_topics[t]));
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");
    fflush(f);
}

/*
 * save the model in the text format
 */

void slda::save_model_text(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "w");
    //print elsewhere!!!
    //fprintf(file, "alpha: %lf\n", alpha);
    fprintf(file, "number of word types: %d\n", num_word_types);
    fprintf(file, "number of authors: %d\n", num_classes);
    fprintf(file, "\nmetrics:\n");

    for (int t  = 0; t < num_word_types; t++)
    {
        fprintf(file, "    - word type: %d\n", t);
        fprintf(file, "      number of topics: %d\n", num_topics[t]);
        fprintf(file, "      size of vocab: %d\n", size_vocab[t]);

        fprintf(file, "      vocab distribution: ["); // in log space
        for (int k = 0; k < num_topics[t]; k++)
        {
            if (k)
            {
                fprintf(file, ", ");
            }
            fprintf(file, "[");
            for (int j = 0; j < size_vocab[t]; j++)
            {
                fprintf(file, "%lf", log_prob_w[t][k][j]);
                if (j < size_vocab[t] - 1)
                {
                    fprintf(file, ", ");
                }
            }
            fprintf(file, "]");
        }
        fprintf(file, "]\n");

        fprintf(file, "      global alphas: [");
        for (int k = 0; k < num_topics[t]; k++)
        {
            fprintf(file, "%lf", as_global[t]->alpha_t[k]);
            if (k < num_topics[t] - 1)
            {
                fprintf(file, ", ");
            }
        }

        fprintf(file, "]\n");

        fprintf(file, "      per author alphas: [");
        for (int a = 0; a < num_classes; a++)
        {
            if (a)
            {
                fprintf(file, ", ");
            }
            fprintf(file, "[");
            for (int k = 0; k < num_topics[t]; k++)
            {
                fprintf(file, "%lf", as[t][a]->alpha_t[k]);
                if (k < num_topics[t] - 1)
                {
                    fprintf(file, ", ");
                }
            }
            fprintf(file, "]");
        }
        fprintf(file, "]\n");

        fprintf(file, "      etas: [");
        for (int i = 0; i < num_classes-1; i++)
        {
            if (i)
            {
                fprintf(file, ", ");
            }
            fprintf(file, "[");

            for (int j = 0; j < num_topics[t]; j++)
            {
                fprintf(file, "%lf", eta[t][i][j]);
                if (j < num_topics[t] - 1)
                {
                    fprintf(file, ", ");
                }
            }
            fprintf(file, "]");
        }
        fprintf(file, "]\n");
    }

    fflush(file);
    fclose(file);
}

/**
   Finish this later
 **/
//
//void slda::free_suffstats(suffstats ** ss, int t)
/*{
   delete [] ss[t]->word_total_ss;

   for (int k = 0; k < num_topics[t]; k ++)
   {
   delete [] ss[t]->word_ss[k];
   }
   delete [] ss[t]->word_ss;

   for (int d = 0; d < num_docs; d ++)
   {
   delete [] ss[t]->z_bar[d].z_bar_m;
   delete [] ss[t]->z_bar[d].z_bar_var;
   }
   delete [] ss[t]->z_bar;
   delete [] ss[t]->labels;
   delete [] ss[t]->tot_labels;

   delete ss[t];
   /}*/

// return to this later

void slda::load_model_initialize_ss(suffstats* ss, corpus * c, int t)
{
    for (int d = 0; d < num_docs; d++)
    {
        document * doc = c->docs[d];
        ss->labels[d] = doc->label;
        ss->tot_labels[doc->label]++;
    }

}
/*
double slda::lda_inference_2(document* doc, double* var_gamma, double** phi,
                           const settings * setting, int t, int a)
{
    int k, n, t; 

    compute posterior dirichlet
    for (k = 0; k < num_topics[t]; k++)
    {
        digamma_gam[k] = digamma(var_gamma[k]);
    }
    var_iter = 0;

    for (n = 0; n < doc->length[t]; n++)
    {
        phisum = 0;
        for (k = 0; k < num_topics[t]; k++)
        {
            phi[n][k] = digamma_gam[k] + log_prob_w[t][k][doc->words[t][n]];

            if (k > 0)
                phisum = log_sum(phisum, phi[n][k]);
            else
                phisum = phi[n][k]; // note, phi is in log space
        }

        for (k = 0; k < num_topics[t]; k++)
        {
            double t = exp(phi[n][k] - phisum);
            var_gamma[k] = var_gamma[k] + doc->counts[t][n]*t;
            digamma_gam[k] = digamma(var_gamma[k]);
        }

        likelihood = lda_compute_likelihood(doc, phi, var_gamma, t,-1);
        assert(!isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;
    }

    delete [] oldphi;
    delete [] digamma_gam;

    return likelihood;
}
*/


double slda::lda_inference(document* doc, double* var_gamma, double** phi,
                           const settings * setting, int t, int a)
{
    int k, n, var_iter;
    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;

    double *oldphi = new double [num_topics[t]];
    double *digamma_gam = new double [num_topics[t]];
    alphas * lda_alphas;

    if (a == -1)
        lda_alphas = as_global[t];
    else
        lda_alphas = as[t][a];

    if (a == -1)
        alphas * lda_alphas = as_global[t];
    else
        alphas * lda_alphas = as[t][a];

    // compute posterior dirichlet
    for (k = 0; k < num_topics[t]; k++)
    {
        // change later to use a local version
        //std::cout << "alpha_t[k]: " << lda_alphas->alpha_t[k] << "\n";
       // std::cout << "doc->total[t]: " << doc->total[t] << "\n";
        //std::cout << "num_topics[t]" << num_topics[t] << "\n";
        var_gamma[k] = lda_alphas->alpha_t[k] + (doc->total[t]/((double) num_topics[t]));
        digamma_gam[k] = digamma(var_gamma[k]);
        for (n = 0; n < doc->length[t]; n++)
            phi[n][k] = 1.0/num_topics[t];
    }
    var_iter = 0;

    while (converged > setting->VAR_CONVERGED && (var_iter < setting->VAR_MAX_ITER || setting->VAR_MAX_ITER == -1))
    {
        var_iter++;
        for (n = 0; n < doc->length[t]; n++)
        {
            phisum = 0;
            for (k = 0; k < num_topics[t]; k++)
            {
                oldphi[k] = phi[n][k];
                phi[n][k] = digamma_gam[k] + log_prob_w[t][k][doc->words[t][n]];

                if (k > 0)
                    phisum = log_sum(phisum, phi[n][k]);
                else
                    phisum = phi[n][k]; // note, phi is in log space
            }

            for (k = 0; k < num_topics[t]; k++)
            {
                phi[n][k] = exp(phi[n][k] - phisum);
                var_gamma[k] = var_gamma[k] + doc->counts[t][n]*(phi[n][k] - oldphi[k]);
                digamma_gam[k] = digamma(var_gamma[k]);
            }
        }

        likelihood = lda_compute_likelihood(doc, phi, var_gamma, t,-1);
        assert(!isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;
    }

    delete [] oldphi;
    delete [] digamma_gam;

    return likelihood;
}

double slda::lda_compute_likelihood(document* doc, double** phi, double* var_gamma,
                                    int t, int a )
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0;
    double *dig = new double [num_topics[t]];
    int k, n;

    alphas * lda_alphas;
    if (a == -1)
        lda_alphas = as_global[t];
    else
        lda_alphas = as[t][a];

    double alpha_sum = lda_alphas->alpha_sum_t;

    for (k = 0; k < num_topics[t]; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha_sum) - lgamma(var_gamma_sum);

    for (k = 0; k < num_topics[t]; k++)
    {
        likelihood += -lgamma(lda_alphas->alpha_t[k]) + (lda_alphas->alpha_t[k] - 1)*(dig[k] - digsum) +
                      lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

        for (n = 0; n < doc->length[t]; n++)
        {
            if (phi[n][k] > 0)
            {
                likelihood += doc->counts[t][n]*(phi[n][k]*((dig[k] - digsum) -
                                                            log(phi[n][k]) + log_prob_w[t][k][doc->words[t][n]]));
            }
        }
    }

    delete [] dig;
    return likelihood;
}


// revisit
void slda::corpus_initialize_ss(suffstats* ss, corpus* c, int t)
{
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
    time_t seed;
    time(&seed);
    gsl_rng_set(rng, (long) seed);
    int k, n, d, j, idx, i, w;

    for (k = 0; k < num_topics[t]; k++)
    {
        for (i = 0; i < NUM_INIT; i++)
        {
            d = (int)(floor(gsl_rng_uniform(rng) * num_docs));
            printf("initialized with document %d\n", d);
            document * doc = c->docs[d];
            for (n = 0; n < doc->length[t]; n++)
            {
                ss->word_ss[k][doc->words[t][n]] += doc->counts[t][n];
            }
        }
        for (w = 0; w < size_vocab[t]; w++)
        {
            ss->word_ss[k][w] = 2*ss->word_ss[k][w] + 5 + gsl_rng_uniform(rng);
            ss->word_total_ss[k] = ss->word_total_ss[k] + ss->word_ss[k][w];
        }
    }


    for (d = 0; d < num_docs; d++)
    {
        document * doc = c->docs[d];
        ss->labels[d] = doc->label;
        ss->tot_labels[doc->label]++;

        double total = 0.0;
        for (k = 0; k < num_topics[t]; k++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }
        for (k = 0; k < num_topics[t]; k++)
        {
            ss->z_bar[d].z_bar_m[k] /= total;
        }
        for (k = 0; k < num_topics[t]; k++)
        {
            for (j = k; j < num_topics[t]; j++)
            {
                idx = map_idx(k, j, num_topics[t]);
                if (j == k)
                    ss->z_bar[d].z_bar_var[idx] = ss->z_bar[d].z_bar_m[k] / (double)(doc->total[t]);
                else
                    ss->z_bar[d].z_bar_var[idx] = 0.0;

                ss->z_bar[d].z_bar_var[idx] -=
                    ss->z_bar[d].z_bar_m[k] * ss->z_bar[d].z_bar_m[j] / (double)(doc->total[t]);
            }
        }

    }
    gsl_rng_free(rng);
}

/*
 * load the model in the binary format
 * ED
 */


void slda::load_model(const char * filename)
{
    FILE * file = NULL;
    std::cout << "READING MODEL FROM " << filename << "\n";
    file = fopen(filename, "rb");
    //fwrite(&epsilon, sizeof (double), 1, file);
    //fwrite(&num_topics[t], sizeof (int), 1, file);
    //fwrite(&size_vocab[t], sizeof (int), 1, file);

    fread(&num_classes, sizeof (int), 1, file);
    fread(&num_word_types, sizeof (int), 1, file);

    num_topics = new int[num_word_types];
    size_vocab = new int[num_word_types];

    // double *** log_prob_w =  new double ** [num_word_types];
    as =  new alphas ** [num_word_types];
    as_global =  new alphas * [num_word_types];


    fread(&epsilon, sizeof (double), 1, file);
    for (int t = 0; t < num_word_types; t++)
    {
        fread(&num_topics[t], sizeof (int), 1, file);
        fread(&size_vocab[t], sizeof (int), 1, file);

        log_prob_w.push_back(std::vector<std::vector<double > >());
        for (int k = 0; k < num_topics[t]; k++)
        {
            log_prob_w[t].push_back(std::vector<double >(size_vocab[t], 0));
            for (int w = 0; w < size_vocab[t]; w++)
                fread(&log_prob_w[t][k][w], sizeof(double), 1, file);
        }

        as_global[t] = new alphas;
        as_global[t]->alpha_t = new double [num_topics[t]];
        fread(as_global[t]->alpha_t, sizeof(double), num_topics[t], file);
        as_global[t]->alpha_sum_t = 0;
        for (int i = 0; i < num_topics[t]; i++)
        {
            as_global[t]->alpha_sum_t += as_global[t]->alpha_t[i];
        }

        as[t] = new alphas *  [num_classes];
        for (int a = 0; a < num_classes; a++)
        {
            as[t][a] = new alphas;
            as[t][a]->alpha_t = new double [num_topics[t]];
            fread(as[t][a]->alpha_t, sizeof(double), num_topics[t], file);
            as[t][a]->alpha_sum_t = 0;
            for (int i = 0; i < num_topics[t]; i++)
            {
                as[t][a]->alpha_sum_t += as[t][a]->alpha_t[i];
            }
        }
        if (num_classes > 1)
        {
            eta.push_back(std::vector<std::vector<double > >());
            for (int i = 0; i < num_classes-1; i++)
            {
                eta[t].push_back(std::vector<double>(num_topics[t], 0));
                for (int k = 0; k < num_topics[t]; k++)
                    fread(&eta[t][i][k], sizeof(double), 1, file);
            }
        }
    }



    fflush(file);
    fclose(file);
}

void slda::save_model(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "wb");
    //fwrite(&epsilon, sizeof (double), 1, file);
    //fwrite(&num_topics[t], sizeof (int), 1, file);
    //fwrite(&size_vocab[t], sizeof (int), 1, file);

    fwrite(&num_classes, sizeof (int), 1, file);
    fwrite(&num_word_types,sizeof (int), 1, file);

    fwrite(&epsilon, sizeof (double), 1, file);
    for (int t = 0; t < num_word_types; t++)
    {
        fwrite(&num_topics[t], sizeof (int), 1, file);
        fwrite(&size_vocab[t], sizeof (int), 1, file);

        for (int k = 0; k < num_topics[t]; k++)
        {
            for (int w = 0; w < size_vocab[t]; w++)
                fwrite(&log_prob_w[t][k][w], sizeof(double),1, file);
        }

        fwrite(as_global[t]->alpha_t, sizeof(double), num_topics[t], file);
        for (int a = 0; a < num_classes; a++)
        {
            fwrite(as[t][a]->alpha_t, sizeof(double), num_topics[t], file);
        }
        if (num_classes > 1)
        {
            for (int i = 0; i < num_classes-1; i++)
            {
                for (int k = 0; k < num_topics[t]; k++)
                    fwrite(&eta[t][i][k], sizeof(double), 1, file);
            }
        }
    }

    fflush(file);
    fclose(file);
}

double get_rho(int i)
{
    return (1/pow(1+i,0.6));
}




