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
#include "opt.h"
#include "slda.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf.h>

using namespace std;


double dr_f(const gsl_vector * x, void * opt_param)
{
    double alpha_sum = 0;
    double output = 0;
    dr_parameter * dr = (dr_parameter *)opt_param;

    for (size_t k = 0; k < x->size; k++ )
    {
        //cout << "LOG P at k: " << gsl_vector_get(dr->log_p,k) << " ";
        //cout << "" << k << "\n";
        alpha_sum += gsl_vector_get(x,k);
        output += -1 * gsl_sf_lngamma(gsl_vector_get(x,k));
        output += (gsl_vector_get(x,k) -1)*gsl_vector_get(dr->log_p,k);
    }
    output+=gsl_sf_lngamma(alpha_sum);
    //cout << "output " << output << "\n";
    return -1*output;
}

/*
 * function to compute the derivatives of function 
 *
 */

void dr_df(const gsl_vector * x, void * opt_param, gsl_vector * df)
{

    double alpha_sum = 0;
    double output = 0;
    dr_parameter * dr = (dr_parameter *)opt_param;


    for (size_t k = 0; k < x->size; k++ )
        alpha_sum += gsl_vector_get(x,k);
    alpha_sum = gsl_sf_psi(alpha_sum);

    for (size_t k = 0; k < x->size; k++ )
    {
        output = gsl_vector_get(dr->log_p,k);
        output -= gsl_sf_psi(gsl_vector_get(x,k));
        output += alpha_sum;
        //cout << "output " << output << "\n";
        gsl_vector_set(df,k,-1*output);
    }
}

void dr_fdf(const gsl_vector * x, void * opt_param, double * f, gsl_vector * df)
{
    *f = dr_f(x, opt_param);
    cout << "f " << *f << "\n";

    dr_df(x,opt_param,df);
}

/*
 * Here the implementation is slightly different from the equations
 * in the paper, we instead use a second-order taylor expansion to approximate
 * the second line in eqaution (6).
 */

double softmax_f(const gsl_vector * x, void * opt_param)
{
    opt_parameter * gsl_param = (opt_parameter *)opt_param;
    double PENALTY = gsl_param->PENALTY;
    slda * model = gsl_param->model;
    std::vector<suffstats *> ss = gsl_param->ss;

    double f, t0, a1 = 0.0, a2 = 0.0;

    int k, d, j, l, idx, t;

    double f_regularization = 0.0;


    for (t = 0; t < model->num_word_types; t++)
        for (l = 0; l < model->num_classes-1; l++)
        {
            for (k = 0; k < model->num_topics[t]; k++)
            {
                // check this out later
                model->eta[t][l][k] = gsl_vector_get(x, model->vec_index(t, l, k));
                f_regularization -= pow(model->eta[t][l][k], 2) * PENALTY/2.0;
            }
        }
    f = 0.0; //log likelihood
    for (d = 0; d < model->num_docs; d++)
    {
        for (t = 0; t < model->num_word_types; t++)
        {
            for (k = 0; k < model->num_topics[t]; k++)
            {
                if (ss[t]->labels[d] < model->num_classes-1)
                {
                    f += model->eta[t][ss[t]->labels[d]][k] * ss[t]->z_bar[d].z_bar_m[k];
                }
            }
        }

        t0 = 0.0; // in log space,  1+exp()+exp()...
        for (l = 0; l < model->num_classes-1; l++)
        {
            a1 = 0.0; // \eta_k^T * \bar{\phi}_d
            a2 = 0.0; // 1 + 0.5 * \eta_k^T * Var(z_bar)\eta_k

            for (t = 0; t < model->num_word_types; t++)
                for (k = 0; k < model->num_topics[t]; k++)
                {
                    a1 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_m[k];

                    for (j = 0; j < model->num_topics[t]; j++)
                    {
                        // WHAT DOES THIS METHOD DO
                        idx = map_idx(k, j, model->num_topics[t]);
                        a2 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_var[idx] * model->eta[t][l][j];
                    }
                }
            a2 = 1.0 + 0.5 * a2;
            t0 = log_sum(t0, a1 + log(a2));
        }
        f -= t0;
    }

    return -(f + f_regularization);
}
void softmax_df(const gsl_vector * x, void * opt_param, gsl_vector * df)
{

    opt_parameter * gsl_param = (opt_parameter *)opt_param;
    double PENALTY = gsl_param->PENALTY;
    slda * model = gsl_param->model;
    std::vector<suffstats *> ss = gsl_param->ss;

    gsl_vector_set_zero(df);
    gsl_vector * df_tmp = gsl_vector_alloc(df->size);

    double a1 = 0.0, a2 = 0.0, g,t0;
    int k, d, j, l, idx, t;


    double ** eta_aux = new double * [model->num_word_types];

    for (t = 0; t < model->num_word_types; t++)
    {
        eta_aux[t] = new double[model->num_topics[t]];
    }


    for (t = 0; t < model->num_word_types; t++)
        for (l = 0; l < model->num_classes-1; l++)
            for (k = 0; k < model->num_topics[t]; k++)
            {
                idx = model->vec_index(t,l,k);
                model->eta[t][l][k] = gsl_vector_get(x, idx);
                g = -PENALTY * model->eta[t][l][k];
                gsl_vector_set(df, idx, g);
            }

    for (d = 0; d < model->num_docs; d++)
    {
        for (t = 0; t < model->num_word_types; t++)
        {
            for (k = 0; k < model->num_topics[t]; k++)
            {
                l = ss[t]->labels[d];
                if (l < model->num_classes-1)
                {
                    idx = model->vec_index(t,l,k);
                    g = gsl_vector_get(df, idx) + ss[t]->z_bar[d].z_bar_m[k];
                    gsl_vector_set(df, idx, g);
                }
            }
        }


        t0 = 0.0; // in log space, 1+exp()+exp()+....
        gsl_vector_memcpy(df_tmp, df);
        gsl_vector_set_zero(df);
        for (l = 0; l < model->num_classes-1; l++)
        {
            for (t = 0; t < model->num_word_types; t++)
            {
                memset(eta_aux[t], 0, sizeof(double)*model->num_topics[t]);
            }

            a1 = 0.0; // \eta_k^T * \bar{\phi}_d
            a2 = 0.0; // 1 + 0.5*\eta_k^T * Var(z_bar)\eta_k
            for (t = 0; t < model->num_word_types; t++)
            {
                for (k = 0; k < model->num_topics[t]; k++)
                {
                    a1 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_m[k];
                    for (j = 0; j < model->num_topics[t]; j++)
                    {
                        idx = map_idx(k, j, model->num_topics[t]);
                        a2 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_var[idx] * model->eta[t][l][j];
                        eta_aux[t][k] += ss[t]->z_bar[d].z_bar_var[idx] * model->eta[t][l][j];
                    }
                }
            }
            a2 = 1.0 + 0.5 * a2;
            t0 = log_sum(t0, a1 + log(a2));

            for (t = 0; t < model->num_word_types; t++)
            {
                for (k = 0; k < model->num_topics[t]; k++)
                {
                    idx = model->vec_index(t, l, k);
                    g =  gsl_vector_get(df, idx) -
                        exp(a1) * (ss[t]->z_bar[d].z_bar_m[k] * a2 + eta_aux[t][k]);
                    gsl_vector_set(df, idx, g);
                }
            }
        }
        gsl_vector_scale(df, exp(-t0));
        gsl_vector_add(df, df_tmp);
    }
    gsl_vector_scale(df, -1.0);
    delete [] eta_aux;
    gsl_vector_free(df_tmp);
}

void softmax_fdf(const gsl_vector * x, void * opt_param, double * f, gsl_vector * df)
{
    opt_parameter * gsl_param = (opt_parameter *)opt_param;
    double PENALTY = gsl_param->PENALTY;
    slda * model = gsl_param->model;
    std::vector<suffstats *> ss = gsl_param->ss;
    gsl_vector_set_zero(df);
    gsl_vector * df_tmp = gsl_vector_alloc(df->size);

    double t0, a1 = 0.0, a2 = 0.0, g;
    int k, d, j, l, idx, t;

    double f_regularization = 0.0;

    double ** eta_aux = new double * [model->num_word_types];
    for (t = 0; t < model->num_word_types; t++)
    {
        eta_aux[t] = new double[model->num_topics[t]];
    }

    for (l = 0; l < model->num_classes-1; l++)
    {
        for (t = 0; t < model->num_word_types; t++)
            for (k = 0; k < model->num_topics[t]; k++)
            {
                model->eta[t][l][k] = gsl_vector_get(x, model->vec_index(t, l, k));
                f_regularization -= pow(model->eta[t][l][k], 2) * PENALTY/2.0;
                idx = model->vec_index(t, l, k);
                g = -PENALTY * model->eta[t][l][k];
                gsl_vector_set(df, idx, g);
            }
    }
    *f = 0.0; //log likelihood
    for (d = 0; d < model->num_docs; d++)
    {
        for (t = 0; t < model->num_word_types; t++)
            for (k = 0; k < model->num_topics[t]; k++)
            {
                l = ss[t]->labels[d];
                if (l < model->num_classes-1)
                {
                    *f += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_m[k];
                    idx = model->vec_index(t, l, k);
                    g = gsl_vector_get(df, idx) + ss[t]->z_bar[d].z_bar_m[k];
                    gsl_vector_set(df, idx, g);
                }
            }
        t0 = 0.0; // in log space,  base class 1+exp()+exp()
        gsl_vector_memcpy(df_tmp, df);
        gsl_vector_set_zero(df);
        for (l = 0; l < model->num_classes-1; l++)
        {
            for (t = 0; t < model->num_word_types; t++)
            {
                memset(eta_aux[t], 0, sizeof(double)*model->num_topics[t]);
            }

            a1 = 0.0; // \eta_k^T * \bar{\phi}_d
            a2 = 0.0; // 1 + 0.5 * \eta_k^T * Var(z_bar)\eta_k

            for (t = 0; t < model->num_word_types; t++)
            {
                for (k = 0; k < model->num_topics[t]; k++)
                {
                    a1 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_m[k];

                    for (j = 0; j < model->num_topics[t]; j++)
                    {
                        idx = map_idx(k, j, model->num_topics[t]);
                        a2 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_var[idx] * model->eta[t][l][j];
                        eta_aux[t][k] += ss[t]->z_bar[d].z_bar_var[idx] * model->eta[t][l][j];
                    }
                }
            }
            a2 = 1.0 + 0.5 * a2;
            t0 = log_sum(t, a1 + log(a2));

            for (t = 0; t < model->num_word_types; t++)
                for (k = 0; k < model->num_topics[t]; k++)
                {
                    idx = model->vec_index(t, l, k);
                    g =  gsl_vector_get(df, idx) -
                        exp(a1) * (ss[t]->z_bar[d].z_bar_m[k] * a2 + eta_aux[t][k]);
                    gsl_vector_set(df, idx, g);
                }
        }
        gsl_vector_scale(df, exp(-t0));
        gsl_vector_add(df, df_tmp);
        *f -= t0;
    }
    gsl_vector_scale(df, -1.0);
    *f = -(*f + f_regularization);
    delete [] eta_aux;
    gsl_vector_free(df_tmp);
}

// samples size vectors from 
int * sample(std::vector<double> prob, int trials, gsl_rng * rng)
{
    int * output = new int[prob.size()];
    //gsl_ran_multinomial(rng, prob.size(), trials, (double []) &prob[0], (double []) output);
    return output;
}

/*
void softmax_df_stoch(const gsl_vector * x, void * opt_param, gsl_vector * df)
   {

    stoch_opt_parameter * gsl_param = (stoch_opt_parameter *)opt_param;
    double PENALTY = gsl_param->PENALTY;
    slda * model = gsl_param->model;
    std::vector<suffstats *> ss = gsl_param->ss;

    std::vector<double> author_prob = gsl_param->author_prob;
    std::vector<double> doc_prob = gsl_param-> doc_prob;
    int author_trials = gsl_param->author_trials;
    int doc_trials = gsl_param->doc_trials;
    gsl_rng * rng = gsl_param->rng;
    int[] stoch_authors;

    if (gsl_param->sample_authors)
         stoch_authors = sample(author_prob, author_trials, rng);
    else
    {
        author_trials = model->num_classes-1;
        stoch_authors = new double[author_trials]
        for (size_t a = 0, a < author_trials; a++)
        {
            stoch_authors[a] = a;
            author_prob[a] = 1/((double)author_trials);
        }
    }

    int[] stoch_docs =  sample(doc_prob, author_trials, rng);

    gsl_vector_set_zero(df);
    gsl_vector * df_tmp = gsl_vector_alloc(df->size);

    double a1 = 0.0, a2 = 0.0, g,t0, dp,lp;
    int k, d, j, l, idx, t, li, di,li;


    double ** eta_aux = new double * [model->num_word_types];

    for (t = 0; t < model->num_word_types; t++) {
        eta_aux[t] = new double[model->num_topics[t]];
    }


    for (t = 0; t < model->num_word_types; t++)
    {
        for (li = 0; li < author_trials; li++)
        {
            l =  stoch_authors[li];
            dp = author_prob[li];

            for (k = 0; k < model->num_topics[t]; k++)
            {
                // scaled for stochastic descent
                idx = model->vec_index(t,l,k);
                model->eta[t][l][k] = gsl_vector_get(x, idx);
                g = (doc_trials/dp)*-PENALTY * model->eta[t][l][k];
                gsl_vector_set(df, idx, g);
            }
        }
    }

    for (di = 0; di < doc_trials; di++)
    {
        d = stoch_docs[di];
        dp = doc_prob[di];

        for (t = 0; t < model->num_word_types; t++) {
            for (k = 0; k < model->num_topics[t]; k++)
            {
                l = ss[t]->labels[d];
                if(l < model.num_classes - 1)
                {
                    // REMEMBER TO SCALE
                    idx = model->vec_index(t,l,k);
                    g = *gsl_vector_get(df, idx) + ss[t]->z_bar[d].z_bar_m[k];
                    gsl_vector_set(df, idx, g);
                }
            }
        }


        t0 = 0.0; // in log space, 1+exp()+exp()+....
        gsl_vector_memcpy(df_tmp, df);
        gsl_vector_set_zero(df);
        for (li = 0; li < stoch_authors.size(); li++)
        {
            l = stoch_authors[li];
            lp = author_prob[li];


            for (t = 0; t < model->num_word_types; t++) {
                memset(eta_aux[t], 0, sizeof(double)*model->num_topics[t]);
            }

            a1 = 0.0; // \eta_k^T * \bar{\phi}_d
            a2 = 0.0; // 1 + 0.5*\eta_k^T * Var(z_bar)\eta_k

            for (t = 0; t < model->num_word_types; t++) {
                
                for (k = 0; k < model->num_topics[t]; k++)
                {
                    
                    a1 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_m[k];
                    for (j = 0; j < model->num_topics[t]; j++)
                    {
                        
                        idx = map_idx(k, j, model->num_topics[t]);
                        a2 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_var[idx] * model->eta[t][l][j];
                        eta_aux[t][k] += ss[t]->z_bar[d].z_bar_var[idx] * model->eta[t][l][j];
                    }
                }

            }

            a2 = 1.0 + 0.5 * a2;
            t0 = log_sum(t0, a1 + log(a2));

            double scale = (doc_trials/dp)*(author_trials/lp);
            for (t = 0; t < model->num_word_types; t++) {
                for (k = 0; k < model->num_topics[t]; k++)
                {
                    //DO SCALING
                    idx = model->vec_index(t, l, k);
                    g =  gsl_vector_get(df, idx) - scale*
                        (exp(a1) * (ss[t]->z_bar[d].z_bar_m[k] * a2 + eta_aux[t][k]));
                    gsl_vector_set(df, idx, g);
                }
            }
        }
        gsl_vector_scale(df, exp(-t0));
        gsl_vector_add(df, df_tmp);
    }
    gsl_vector_scale(df, -1.0);
    delete [] eta_aux;
    gsl_vector_free(df_tmp);
   }

void softmax_fdf_stoch(const gsl_vector * x, void * opt_param, double * f, gsl_vector * df)
{
        softmax_f(x, opt_param);
        softmax_df_stoch(x, opt_param, df)
}
*/

/**
   double softmax_f_stoch(const gsl_vector * x, void * opt_param)
   {
    stoch_opt_parameter * gsl_param = (stoch_opt_parameter *)opt_param;
    double PENALTY = gsl_param->PENALTY;
    slda * model = gsl_param->model;
    std::vector<suffstats *> ss = gsl_param->ss;

    std::vector<int> stoch_authors = gsl_param->stoch_authors;
    std::vector<int> stoch_docs =  gsl_param->stoch_docs;

    std::vector<double> author_prob = gsl_param->author_prob;
    std::vector<double> doc_prob = gsl_param-> doc_prob;

    double f, t0, a1 = 0.0, a2 = 0.0, dp, lp;

    int k, d, j, l, idx, t, li, di;

    double f_regularization = 0.0;


    for (t = 0; t < model->num_word_types; t++)
    {
        for (li = 0; li < model->num_classes-1; li++)
        {
            l =  stoch_authors[li];
            ld = author_prob[li];
            for (k = 0; k < model->num_topics[t]; k++)
            {
                // check this out later

                model->eta[t][l][k] = gsl_vector_get(x,
                    model->vec_index2(t, li, k,stoch_authors.size()));
                f_regularization -= pow(model->eta[t][l][k], 2) * PENALTY/2.0;
            }
        }
    }
    f = 0.0; //log likelihood
    for (di = 0; di < stoch_docs.size(); di++)
    {
        d = stoch_docs[di];
        di = doc_prob[di];

        for (t = 0; t < model->num_word_types; t++) {
            for (k = 0; k < model->num_topics[t]; k++)
            {
                if (ss[t]->labels[d] < model->num_classes-1)
                {
                    f += model->eta[t][ss[t]->labels[d]][k] * ss[t]->z_bar[d].z_bar_m[k];
                }
            }
        }

        t0 = 0.0; // in log space,  1+exp()+exp()...
        for (li = 0; li < stoch_authors.size(); li++)
        {
            l =  stoch_authors[li];
            ld = author_prob[li];

            a1 = 0.0; // \eta_k^T * \bar{\phi}_d
            a2 = 0.0; // 1 + 0.5 * \eta_k^T * Var(z_bar)\eta_k

            for (t = 0; t < model->num_word_types; t++)
            for (k = 0; k < model->num_topics[t]; k++)
            {
                a1 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_m[k];

                for (j = 0; j < model->num_topics[t]; j++)
                {
                    // WHAT DOES THIS METHOD DO
                    idx = map_idx(k, j, model->num_topics[t]);
                    a2 += model->eta[t][l][k] * ss[t]->z_bar[d].z_bar_var[idx] * model->eta[t][l][j];
                }
            }
            a2 = 1.0 + 0.5 * a2;
            t0 = log_sum(t0, a1 + log(a2));
        }
        f -= t0;
    }

    return -(f + f_regularization);
   }
   **/


