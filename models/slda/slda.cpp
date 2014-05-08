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
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
//# "fitDirichlet"

#include <vector>
using namespace std;

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

void slda::init(double epsilon2, int * num_topics_,
                const corpus * c)
{
    num_topics = num_topics_;

    docs_per = c->docsPer;
    docAuthors = c->docAuthors;

    num_word_types = c->num_word_types;
    num_docs = c->num_docs;

    size_vocab = c->size_vocab;
    num_classes = c->num_classes;

    init_alpha(epsilon2);

    // iterate through each type of word
    for (int t = 0; t < num_word_types; t ++)
    {

        log_prob_w.push_back(vector<vector<double > >());
        eta.push_back(vector<vector<double > >());

        // go through each topic for a word type
        for (int k = 0; k < num_topics[t]; k++)
        {
            log_prob_w[t].push_back(vector<double>(size_vocab[t], 0));
        }

        for (int i = 0; i < num_classes-1; i ++)
        {
            eta[t].push_back(vector<double>(num_topics[t], 0));
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

    vector<double>::iterator it;
    it = ss->word_total_ss.begin();
    for (int i = 0; i < num_topics[t]; i ++)
        ss->word_total_ss.push_back(0);

    //ss->word_total_ss.insert(it,num_topics[t],0);
    //ss->word_ss = new double * [num_topics[t]];

    for (int k = 0; k < num_topics[t]; k ++)
    {
        //ss->word_ss[k] = new double [size_vocab[t]];
        //memset(ss->word_ss[k], 0, sizeof(double)*size_vocab[t]);
        ss->word_ss.push_back(vector<double>(size_vocab[t],0));
    }

    int num_var_entries = num_topics[t]*(num_topics[t]+1)/2;
    ss->z_bar = new z_stat [num_docs];
    for (int d = 0; d < num_docs; d ++)
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


/*
 * initialize the sufficient statistics with zeros
 */

void slda::zero_initialize_ss(suffstats * ss, int t)
{
    //memset(ss->word_total_ss, 0, sizeof(double)*num_topics[t]);
    ss->word_total_ss.assign(num_topics[t],0);
    for (int k = 0; k < num_topics[t]; k ++)
    {
        ss->word_ss[k].assign(size_vocab[t],0);
        //memset(ss->word_ss[k], 0, sizeof(double)*size_vocab[t]);
    }

    int num_var_entries = num_topics[t]*(num_topics[t]+1)/2;

    for (int d = 0; d < num_docs; d ++)
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

    for (d = 0; d < num_docs; d ++)
    {
        document * doc = c->docs[d];
        ss->labels[d] = doc->label; // in general, doc label = author
        ss->tot_labels[doc->label] ++;

        double total = 0.0;
        for (k = 0; k < num_topics[t]; k ++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }
        for (k = 0; k < num_topics[t]; k ++)
        {
            ss->z_bar[d].z_bar_m[k] /= total;
        }
        for (k = 0; k < num_topics[t]; k ++)
        {
            for (j = k; j < num_topics[t]; j ++)
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
void slda::updatePrior(double *** var_gamma)
{
    for (int t = 0; t < num_word_types; t++) {
        for (int a = 0; a < num_classes; a++ )
        {
            gsl_matrix * mat = gsl_matrix_alloc(docs_per[a],num_topics[t]);

            for (int d = 0; d < docs_per[a]; d++)
            {
                int dd = getDoc(a,d);
                for (int k = 0; k < num_topics[t]; k++)
                    gsl_matrix_set(mat, d, k, var_gamma[dd][t][k]);
                fitDirichlet(mat);
            }
        }
    }
}

void slda::fitDirichlet(gsl_matrix * mat)
{return;}

void slda::globalPrior(double *** var_gamma)
{
    for (int t = 0; t < num_word_types; t++)
        for (int d = 0; d < num_docs; d++ )
        {
            gsl_matrix * mat = gsl_matrix_alloc(num_docs,num_topics[t]);
            for (int k = 0; k < num_topics[t]; k++)
                gsl_matrix_set(mat, d,k, var_gamma[t][d][k]);
            fitDirichlet(mat);
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



    printf("initializing ...\n");

    std::vector<suffstats * > ss;

    for (t = 0; t < num_word_types; t++)
    {
        ss.push_back(new_suffstats(t));
        //if (strcmp(start, "seeded") == 0)
        //{
        //    corpus_initialize_ss(ss[t], c);
        //    mle(ss[t], 0, setting);
        //}
        //else if (strcmp(start, "random") == 0)
        //{
        random_initialize_ss(ss[t], c,t );
        //}
        //else
        //{
        //   load_model(start);
        //  load_model_initialize_ss(ss[t], c);
        //}
    }
    mle(ss, 0, setting);

    // allocate variational parameters
    double *** var_gamma, *** phi, ** lambda;
    var_gamma = new double ** [c->num_docs];
    for (d = 0; d < c->num_docs; d++) {
        var_gamma[d] = new double * [num_word_types];
    }

    phi = new double ** [num_word_types];
    for (t = 0; t < num_word_types; t++)
    {
        phi[t] = new double * [max_length[t]];
        for (d = 0; d < c->num_docs; d++)
        {
            var_gamma[d][t] = new double [num_topics[t]];
            var_gamma[d][t][0] = 0;
        }

        for (n = 0; n < max_length[t]; n++)
            phi[t][n] = new double [num_topics[t]];
    }

// COME BACK LATER
    FILE * likelihood_file = NULL;
    sprintf(filename, "%s/likelihood.dat", directory);
    likelihood_file = fopen(filename, "w");

    int ETA_UPDATE = 0;

    i = 0;
    // CHECK THIS LATER
    while (((converged < 0) || (converged > setting->EM_CONVERGED) || (i <= LDA_INIT_MAX+2)) && (i <= setting->EM_MAX_ITER))
    {
        printf("**** em iteration %d ****\n", ++i);
        likelihood = 0;
        for (t = 0; t < num_word_types; t++)
        {
            zero_initialize_ss(ss[t],t);
        }
        if (i > LDA_INIT_MAX) ETA_UPDATE = 1;
        // e-step
        printf("**** e-step ****\n");
        for (d = 0; d < c->num_docs; d++)
        {
            if ((d % 100) == 0) printf("document %d\n", d);
            std::cout << d << "\n";
            if (true)
                likelihood += slda_inference(c->docs[d], var_gamma[d], phi, as,  d, setting);
            for (t=0; t< num_word_types; t++)
                likelihood += doc_e_step(c->docs[d], var_gamma[d][t], phi[t], ss[t], ETA_UPDATE, d,  t, setting);
        }

        updatePrior(var_gamma);

        printf("likelihood: %10.10f\n", likelihood);
        // m-st
        printf("**** m-step ****\n");

        mle(ss, ETA_UPDATE, setting);

        // check for convergence
        converged = fabs((likelihood_old - likelihood) / (likelihood_old));
        //if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2;
        likelihood_old = likelihood;

        // output model and likelihood
        fprintf(likelihood_file, "%10.10f\t%5.5e\n", likelihood, converged);
        fflush(likelihood_file);
        if ((i % LAG) == 0)
        {
            sprintf(filename, "%s/%03d.model", directory, i);
            save_model(filename);
            sprintf(filename, "%s/%03d.gamma", directory, i);
            save_gamma(filename, var_gamma, c->num_docs);

            for (t =0; t < num_word_types; t++)
            {
                sprintf(filename, "%s/%03d.model.text%i", directory, i,t);
                save_model_text(filename,t);
            }
        }
    }



    // output the final model
    sprintf(filename, "%s/final.model", directory);
    save_model(filename);
    sprintf(filename, "%s/final.gamma", directory);
    //save_gamma(filename, var_gamma, c->num_docs);

    for (t =0; t < num_word_types; t++)
    {
        sprintf(filename, "%s/final.model%i.text", directory, t);
        save_model_text(filename, t);
    }

    fclose(likelihood_file);
    FILE * w_asgn_file = NULL;
    sprintf(filename, "%s/word-assignments.dat", directory);
    w_asgn_file = fopen(filename, "w");
    for (d = 0; d < c->num_docs; d ++)
    {
        //final inference
        if ((d % 100) == 0) printf("final e step document %d\n", d);

        likelihood += slda_inference(c->docs[d], var_gamma[d], phi, as, d ,setting);

        write_word_assignment(w_asgn_file, c->docs[d], phi);

    }
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
    for (size_t ti = 0; ti < t; ti++) {
        output += (num_classes - 1) * (num_topics[ti]);
    }

    output += l * num_topics[t];

    output += k;

    return output;
}

void slda::mle(vector<suffstats *> ss, int eta_update, const settings * setting)
{
    int k, w, t;

    for (t = 0; t < num_word_types; t++)
        for (k = 0; k < num_topics[t]; k++)
        {
            for (w = 0; w < size_vocab[t]; w++)
            {
                if (ss[t]->word_ss[k][w] > 0) {
                    log_prob_w[t][k][w] = (double)log(ss[t]->word_ss[k][w]) - log(ss[t]->word_total_ss[k]);
                } else {
                    log_prob_w[t][k][w] = -100.0;
                }
            }
        }
    if (eta_update == 0) { return; }

    //the label part goes here
    printf("Maximizing ...\n");
	double f = 0.0;
	int status;
	int opt_iter;

    // the total length of all the eta parameters
    int opt_size = 0;
    for (t = 0; t< num_word_types; t++)
        opt_size += num_topics[t];
    opt_size = opt_size * (num_classes-1);

    std::cout << opt_size << "\n";

	int l;

	opt_parameter param;
	param.ss = ss;
	param.model = this;
	param.PENALTY = setting->PENALTY;

	const gsl_multimin_fdfminimizer_type * T;
	gsl_multimin_fdfminimizer * s;
	gsl_vector * x;
	gsl_multimin_function_fdf opt_fun;
	opt_fun.f = &softmax_f;
	opt_fun.df = &softmax_df;
	opt_fun.fdf = &softmax_fdf;
	opt_fun.n = opt_size;
	opt_fun.params = (void*)(&param);
	x = gsl_vector_alloc(opt_size);


    // allocate the long vector of eta values to be optimized
    for (t = 0; t < num_word_types; t++)
        for (l = 0; l < num_classes-1; l ++)
            for (k = 0; k < num_topics[t]; k ++)
                gsl_vector_set(x, vec_index(t,l,k), eta[t][l][k]);

	T = gsl_multimin_fdfminimizer_vector_bfgs;
	s = gsl_multimin_fdfminimizer_alloc(T, opt_size);
	gsl_multimin_fdfminimizer_set(s, &opt_fun, x, 0.02, 1e-4);

	opt_iter = 0;
	do
	{
		opt_iter ++;
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
        for (l = 0; l < num_classes-1; l ++)
            for (k = 0; k < num_topics[t]; k ++)
                eta[t][l][k] = gsl_vector_get(s->x, vec_index(t,l,k));

	gsl_multimin_fdfminimizer_free (s);
	gsl_vector_free (x);

	printf("final f: %f\n", f);
}

double slda::doc_e_step(document* doc, double* gamma, double** phi,
                        suffstats * ss, int eta_update, int _docNum, int t,const settings * setting)
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

    for (n = 0; n < doc->length[t]; n++)
    {
        for (k = 0; k < num_topics[t]; k++)
        {
            ss->word_ss[k][doc->words[t][n]] += doc->counts[t][n]*phi[n][k];
            ss->word_total_ss[k] += doc->counts[t][n]*phi[n][k];

            //statistics for each document of the supervised part
            ss->z_bar[d].z_bar_m[k] += doc->counts[t][n] * phi[n][k]; //mean
            for (i = k; i < num_topics[t]; i ++) //variance
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
    for (i = 0; i < num_topics[t]*(num_topics[t]+1)/2; i ++)
    {
        ss->z_bar[d].z_bar_var[i] /= (double)(doc->total[t] * doc->total[t]);
    }

    //ss->num_docs = ss->num_docs + 1; //because we need it for store statistics for each docs

    return (likelihood);
}



double slda::slda_compute_likelihood(document* doc, double*** phi, double** var_gamma)
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


    for (t=0; t< num_word_types; t++)
    {
        // computes digamma terms for the gamma variation parameters
        var_gamma_sum = 0;
        for (k = 0; k < num_topics[t]; k++)
        {
            dig[t][k] = digamma(var_gamma[t][k]);
            var_gamma_sum += var_gamma[t][k];
        }
        digsum[t] = digamma(var_gamma_sum);
        //computes contribution of Dirichlet paraemter sums to log likelihood
        likelihood += lgamma((as[t][a]->alpha_sum_t)) - lgamma(var_gamma_sum);
    }

    //t0 stors likelihood contributions
    t0 = 0.0;
    for (t= 0; t < num_word_types; t++)
    {
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
        likelihood += t0 / (double)(doc->total[t]); 	//eta_k*\bar{\phi}
    }

    t0 = 1.0; //the class model->num_classes-1
    for (l = 0; l < num_classes-1; l ++)
    {
        t1 = 1.0;
        for (n = 0; n < doc->length[t]; n ++)
        {
            t2 = 0.0;
            for (k = 0; k < num_topics[t]; k ++)
            {
                t2 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
            }
            t1 *= t2;
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

double slda::slda_inference(document* doc, double ** var_gamma, double *** phi,
                            alphas *** as, int d, const settings * setting)
{
    int k, n, var_iter, l,t ;
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
            var_gamma[t][k] = 0;
            var_gamma[t][k] = as[t][a]->alpha_t[k] + (doc->total[t]/((double) num_topics[t]));
            digamma_gam[t][k] = digamma(var_gamma[t][k]);
            for (n = 0; n < doc->length[t]; n++)
                phi[t][n][k] = 1.0/(double)(num_topics[t]);
        }
    }


    for (l = 0; l < num_classes-1; l ++)
    {
        sf_aux[l] = 1.0; // the quantity for equation 6 of each class

        for (t = 0; t < num_word_types; t++)
            for (n = 0; n < doc->length[t]; n ++)
            {
                t0 = 0.0;
                for (k = 0; k < num_topics[t]; k ++)
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
                for (l = 0; l < num_classes-1; l ++)
                {
                    t0 = 0.0;
                    for (k = 0; k < num_topics[t]; k ++)
                    {
                        t0 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                    }
                    sf_aux[l] /= t0; //take out word n

                    for (k = 0; k < num_topics[t]; k ++)
                    {
                        //h in the paper
                        sf_params[k] += sf_aux[l]*exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                    }
                }

                //
                for (k = 0; k < num_topics[t]; k++)
                {
                    oldphi[k] = phi[t][n][k];
                }
                for (fp_iter = 0; fp_iter < FP_MAX_ITER; fp_iter ++) //fixed point update
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
                for (l = 0; l < num_classes-1; l ++)
                {
                    t0 = 0.0;
                    for (k = 0; k < num_topics[t]; k ++)
                    {
                        t0 += phi[t][n][k] * exp(eta[t][l][k] * doc->counts[t][n]/(double)(doc->total[t]));
                    }
                    sf_aux[l] *= t0;
                }
                for (k = 0; k < num_topics[t]; k++)
                {
                    var_gamma[t][k] = var_gamma[t][k] + doc->counts[t][n]*(phi[t][n][k] - oldphi[k]);
                    digamma_gam[t][k] = digamma(var_gamma[t][k]);
                }
            }

        }
        likelihood = slda_compute_likelihood(doc, phi, var_gamma);
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

//fix later
void slda::infer_only(corpus * c, const settings * setting, const char * directory)
{
    int i, k, d, n, t;
    double ** phi_m;
    double base_score, score;
    int label;
    int num_correct = 0;

    char filename[100];
    int * max_length = c->max_corpus_length();
    double ***var_gamma, ***phi;
    double likelihood, likelihood_old = 0, converged = 1;

    // allocate variational parameters
    var_gamma = new double ** [c->num_docs];
    for (t = 0; t < num_word_types;t++)
        var_gamma[t] = new double * [num_word_types];

    phi = new double ** [num_word_types];
    phi_m = new double * [num_word_types];

    for (t = 0; t < num_word_types; t++)
    {
        phi_m[t] = new double [num_topics[t]];

        phi[t] = new double * [max_length[t]];
        for (d = 0; d < c->num_docs; d++)
            var_gamma[d][t] = new double [num_topics[t]];

        for (n = 0; n < max_length[t]; n++)
            phi[t][n] = new double [num_topics[t]];
    }

    FILE * likelihood_file = NULL;
    sprintf(filename, "%s/inf-likelihood.dat", directory);
    likelihood_file = fopen(filename, "w");
    FILE * inf_label_file = NULL;
    sprintf(filename, "%s/inf-labels.dat", directory);
    inf_label_file = fopen(filename, "w");

    for (d = 0; d < c->num_docs; d++)
    {
        if ((d % 100) == 0)
            printf("document %d\n", d);

        document * doc = c->docs[d];
        likelihood = 0;
        for (t = 0; t < num_word_types; t ++)
        {

            likelihood += lda_inference(doc, var_gamma[d][t], phi[t], setting,t);

            memset(phi_m[t], 0, sizeof(double)*num_topics[t]); //zero_initialize
            for (n = 0; n < doc->length[t]; n++)
            {
                for (k = 0; k < num_topics[t]; k ++)
                {
                    phi_m[t][k] += doc->counts[t][n] * phi[t][n][k];
                }
            }
            for (k = 0; k < num_topics[t]; k ++)
            {
                phi_m[t][k] /= (double)(doc->total[t]);
            }
        }

        //do classification
        label = num_classes-1;
        base_score = 0.0;
        for (i = 0; i < num_classes-1; i ++)
        {
            score = 0.0;

            for (t = 0; t < num_word_types; t ++)
                for (k = 0; k < num_topics[t]; k ++)
                    score += eta[t][i][k] * phi_m[t][k];

            if (score > base_score)
            {
                base_score = score;
                label = i;
            }
        }
        if (label == doc->label)
            num_correct ++;

        fprintf(likelihood_file, "%5.5f\n", likelihood);
        fprintf(inf_label_file, "%d\n", label);
    }

    printf("average accuracy: %.3f\n", (double)num_correct / (double) c->num_docs);


    sprintf(filename, "%s/inf-gamma.dat", directory);
    save_gamma(filename, var_gamma, c->num_docs);

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

void slda::save_model_text(const char * filename, int t)
{
    FILE * file = NULL;
    file = fopen(filename, "w");
    //print elsewhere!!!
    //fprintf(file, "alpha: %lf\n", alpha);
    fprintf(file, "number of topics: %d\n", num_topics[t]);
    fprintf(file, "size of vocab: %d\n", size_vocab[t]);
    fprintf(file, "number of classes: %d\n", num_classes);

    fprintf(file, "betas: \n"); // in log space
    for (int k = 0; k < num_topics[t]; k++)
    {
        for (int j = 0; j < size_vocab[t]; j ++)
        {
            fprintf(file, "%lf ", log_prob_w[t][k][j]);
        }
        fprintf(file, "\n");
    }
    if (num_classes > 1)
    {
        fprintf(file, "etas: \n");
        for (int i = 0; i < num_classes-1; i ++)
        {
            for (int j = 0; j < num_topics[t]; j ++)
            {
                fprintf(file, "%lf ", eta[t][i][j]);
            }
            fprintf(file, "\n");
        }
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
    for (int d = 0; d < num_docs; d ++)
    {
        document * doc = c->docs[d];
        ss->labels[d] = doc->label;
        ss->tot_labels[doc->label] ++;
    }
}



double slda::lda_inference(document* doc, double* var_gamma, double** phi, const settings * setting, int t)
{
    int k, n, var_iter;
    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;

    double *oldphi = new double [num_topics[t]];
    double *digamma_gam = new double [num_topics[t]];

    // compute posterior dirichlet
    for (k = 0; k < num_topics[t]; k++)
    {
        // change later to use a local version
        var_gamma[k] = as_global[t]->alpha_t[k] + (doc->total[t]/((double) num_topics[t]));
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

        likelihood = lda_compute_likelihood(doc, phi, var_gamma, t);
        assert(!isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;
    }

    delete [] oldphi;
    delete [] digamma_gam;

    return likelihood;
}

double slda::lda_compute_likelihood(document* doc, double** phi, double* var_gamma, int t)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0;
    double *dig = new double [num_topics[t]];
    int k, n;
    double alpha_sum = as_global[t]->alpha_sum_t;
    for (k = 0; k < num_topics[t]; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha_sum) - lgamma(var_gamma_sum);

    for (k = 0; k < num_topics[t]; k++)
    {
        likelihood += - lgamma(as_global[t]->alpha_t[k]) + (as_global[t]->alpha_t[k] - 1)*(dig[k] - digsum) +
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


    for (d = 0; d < num_docs; d ++)
    {
        document * doc = c->docs[d];
        ss->labels[d] = doc->label;
        ss->tot_labels[doc->label] ++;

        double total = 0.0;
        for (k = 0; k < num_topics[t]; k ++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }
        for (k = 0; k < num_topics[t]; k ++)
        {
            ss->z_bar[d].z_bar_m[k] /= total;
        }
        for (k = 0; k < num_topics[t]; k ++)
        {
            for (j = k; j < num_topics[t]; j ++)
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
 *ED
 */


void slda::load_model(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "rb");
    //fwrite(&epsilon, sizeof (double), 1, file);
    //fwrite(&num_topics[t], sizeof (int), 1, file);
    //fwrite(&size_vocab[t], sizeof (int), 1, file);

    fread(&num_classes, sizeof (int), 1, file);
    fread(&num_word_types, sizeof (int), 1, file);

    int * num_topics = new int[num_word_types];
    int * size_vocab = new int[num_word_types];

    // double *** log_prob_w =  new double ** [num_word_types];
    alphas *** as =  new alphas ** [num_word_types];
    alphas ** as_global =  new alphas * [num_word_types];


    for (int t = 0; t < num_word_types; t++)
    {
        fread(&epsilon, sizeof (double), 1, file);
        fread(&num_topics[t], sizeof (int), 1, file);
        fread(&size_vocab[t], sizeof (int), 1, file);

        for (int k = 0; k < num_topics[t]; k++)
        {
            // fread(log_prob_w[t][k], sizeof(double), size_vocab[t], file);
        }

        fread(as_global[t]->alpha_t, sizeof(double), num_topics[t], file);
        for (int a = 0; a < num_classes; a++)
        {
            fread(as[t][a]->alpha_t, sizeof(double), num_topics[t], file);
        }
        if (num_classes > 1)
        {
            for (int i = 0; i < num_classes-1; i ++)
            {
                // fread(eta[t][i], sizeof(double), num_topics[t], file);
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

    for (int t = 0; t < num_word_types; t++)
    {
        fwrite(&epsilon, sizeof (double), 1, file);
        fwrite(&num_topics[t], sizeof (int), 1, file);
        fwrite(&size_vocab[t], sizeof (int), 1, file);

        for (int k = 0; k < num_topics[t]; k++)
        {
            // fwrite(log_prob_w[t][k], sizeof(double), size_vocab[t], file);
        }

        fwrite(as_global[t]->alpha_t, sizeof(double), num_topics[t], file);
        for (int a = 0; a < num_classes; a++)
        {
            fwrite(as[t][a]->alpha_t, sizeof(double), num_topics[t], file);
        }
        if (num_classes > 1)
        {
            for (int i = 0; i < num_classes-1; i ++)
            {
                // fwrite(eta[t][i], sizeof(double), num_topics[t], file);
            }
        }
    }

    fflush(file);
    fclose(file);
}
/**
   void slda::load_model_alpha(const char * filename)
   {

   }
**/


/**
   void slda::save_model(const char * filename, int t)
   {
   FILE * file = NULL;
   file = fopen(filename, "wb");
   //fwrite(&epsilon, sizeof (double), 1, file);
   fwrite(&num_topics[t], sizeof (int), 1, file);
   fwrite(&size_vocab[t], sizeof (int), 1, file);
   fwrite(&num_classes, sizeof (int), 1, file);

   for (int k = 0; k < num_topics[t]; k++)
   {
   fwrite(log_prob_w[t][k], sizeof(double), size_vocab[t], file);
   }
   if (num_classes > 1)
   {
   for (int i = 0; i < num_classes-1; i ++)
   {
   fwrite(eta[i], sizeof(double), num_topics[t], file);
   }
   }

   fflush(file);
   fclose(file);
   }
**/
