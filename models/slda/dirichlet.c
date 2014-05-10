/* Barebones implementation of dirichlet MLE using GSL
 * Based on Python code by Eric Suh: https://github.com/ericsuh/dirichlet,
 * which is in turn based on Thomas P. Minka's Fastfit MATLAB code
 *
 * Author: David Dohan
 */
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_exp.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include "opt.h"
#include <gsl/gsl_multimin.h>


double vec_sum(gsl_vector *v) {
    /* Returns vector sum */
    return v->size * gsl_stats_mean(v->data, v->stride, v->size);
}

void vec_print(gsl_vector *v) {
    for (size_t i = 0; i < v->size; i++) {
        printf("%f\t", v->data[i]);
    }
    printf("\n");
}

void mat_print(gsl_matrix *m) {
    for (size_t i = 0; i < m->size1; i++) {
        gsl_vector_view vv = gsl_matrix_row(m, i);
        vec_print(&vv.vector);
    }
}

gsl_vector *mat_col_mean(gsl_matrix *m) {
    /* Returns a vector with column means of given matrix */
    gsl_vector *v = gsl_vector_alloc(m->size2);
    for (size_t i = 0; i < m->size2; i++)
    {
        gsl_vector_view vv = gsl_matrix_column(m, i);
        gsl_vector_set(v, i, gsl_stats_mean(vv.vector.data, vv.vector.stride, vv.vector.size));
    }
    return v;
}

// provides a smoothed column mean
gsl_vector *mat_col_mean_s(gsl_matrix *m, gsl_vector *w, double weight) {
    /* Returns a vector with column means of given matrix */
    gsl_vector *v = gsl_vector_alloc(m->size2);
    assert(weight >= 0);

    for (size_t i = 0; i < m->size2; i++)
    {
        gsl_vector_view vv = gsl_matrix_column(m, i);
        gsl_vector_set(v, i, gsl_stats_mean(vv.vector.data, vv.vector.stride, vv.vector.size));
    }

    gsl_vector_scale(v, ((double)m->size2)/(m->size2 + weight));
    gsl_vector_scale(w, (weight)/(m->size2 + weight));

    gsl_vector_add(v, w);


    return v;
}

gsl_matrix *mat_elementwise_func(gsl_matrix *m, double (*f)(double)) {
    /* Returns a matrix copy with given function applied to each element */
    gsl_matrix *mm = gsl_matrix_alloc(m->size1, m->size2);
    for (size_t i = 0; i < m->size1; i++) {
        for (size_t j = 0; j < m->size2; j++) {
            gsl_matrix_set(mm, i, j, f(gsl_matrix_get(m, i, j)));
        }
    }
    return mm;
}

gsl_vector *vec_elementwise_func(gsl_vector *v, double (*f)(double)) {
    /* Returns a vector copy with given function applied to each element */
    gsl_vector *vv = gsl_vector_alloc(v->size);
    for (size_t i = 0; i < v->size; i++) {
        vv->data[i] = f(v->data[i]);
    }
    return vv;
}

double _ipsi(double y) {
    /* Inverse psi with Newton's method*/
    double tol = 1.48e-10;
    size_t maxiter = 30;
    double euler = -1 * gsl_sf_psi(1);

    double x0;
    if (y >= -2.22) {
        x0 = gsl_sf_exp(y) + 0.5;
    } else {
        x0 = -1 / (y + euler);
    }

    for (size_t i = 0; i < maxiter; i++) {
        double x1 = x0 - (gsl_sf_psi(x0) - y) / gsl_sf_psi_1(x0);
        if (fabs(x1 - x0) < tol) {
            return x1;
        }
        x0 = x1;
    }
    return x0;
}


gsl_vector *_init_a(gsl_matrix *D) {
    /* Initial guess for Dirichlet parameters */
    // E = D.mean(axis=0)
    gsl_vector *E = mat_col_mean(D);

    // E2 = (D**2).mean(axis=0)
    gsl_matrix *D_squared = mat_elementwise_func(D, gsl_pow_2);
    gsl_vector *E2 = mat_col_mean(D_squared);
    gsl_matrix_free(D_squared);

    // return ((E[0] - E2[0])/(E2[0]-E[0]**2)) * E
    double scale = (E->data[0] - E2->data[0]) / (E2->data[0] - gsl_pow_2(E->data[0]));
    gsl_vector_scale(E, scale);
    gsl_vector_free(E2);

    return E;
}

// smoothed version of initialize
gsl_vector *_init_a_s(gsl_matrix *D, gsl_vector * w, double weight) {
    /* Initial guess for Dirichlet parameters */
    // E = D.mean(axis=0)

    gsl_vector *E = mat_col_mean_s(D, w, weight);

    // E2 = (D**2).mean(axis=0)
    gsl_matrix *D_squared = mat_elementwise_func(D, gsl_pow_2);
    gsl_vector *w_squared = vec_elementwise_func(w, gsl_pow_2);

    gsl_vector *E2 = mat_col_mean_s(D_squared, w_squared, weight);
    gsl_matrix_free(D_squared);

    // return ((E[0] - E2[0])/(E2[0]-E[0]**2)) * E
    double scale = (E->data[0] - E2->data[0]) / (E2->data[0] - gsl_pow_2(E->data[0]));
    gsl_vector_scale(E, scale);
    gsl_vector_free(E2);
    return E;
}

gsl_vector *log_col_mean(gsl_matrix *D) {
    /* Take log of all elements and return vector of column means */
    gsl_matrix *m = mat_elementwise_func(D, gsl_sf_log);
    gsl_vector *logm = mat_col_mean(m);
    gsl_matrix_free(m);
    return logm;
}

gsl_vector *log_col_mean_s(gsl_matrix *D, gsl_vector * w, double weight) {
    /* Take log of all elements and return vector of column means */
    gsl_matrix *m = mat_elementwise_func(D, gsl_sf_log);
    gsl_vector *log_w = vec_elementwise_func(w, gsl_sf_log);
    gsl_vector *logm = mat_col_mean_s(m,log_w, weight);
    gsl_matrix_free(m);
    return logm;
}

double loglikelihood(gsl_matrix *D, gsl_vector *a) {
    /*
     * D: N x K matrix
     *    N is number of observations
     *    K is number of Dirichlet parameters
     * a: K parameters of dirichlet distribution
     */
    // logp = log(D).mean(axis=0)
    gsl_vector *logp = log_col_mean(D);

    // gammaln(a.sum())
    double gamma_sum = gsl_sf_lngamma(vec_sum(a));

    // gammaln(a).sum()
    double sum_of_gamma = 0;
    for (size_t i = 0; i < a->size; i++) {
        sum_of_gamma += gsl_sf_lngamma(a->data[i]);
    }

    // ((a-1) * logp).sum()
    double m1_logp = 0;
    for (size_t i = 0; i < a->size; i++) {
        m1_logp += (a->data[i] - 1) * logp->data[i];
    }
    gsl_vector_free(logp);

    // return N*(gammaln(a.sum()) - gammaln(a).sum() + ((a - 1)*logp).sum())
    return D->size1 * (gamma_sum - sum_of_gamma + m1_logp);

}

double loglikelihood_s(gsl_matrix *D, gsl_vector *a, gsl_vector * w, double weight) {
    /*
     * D: N x K matrix
     *    N is number of observations
     *    K is number of Dirichlet parameters
     * a: K parameters of dirichlet distribution
     */
    // logp = log(D).mean(axis=0)
    gsl_vector *logp = log_col_mean_s(D, w, weight);

    // gammaln(a.sum())
    double gamma_sum = gsl_sf_lngamma(vec_sum(a));

    // gammaln(a).sum()
    double sum_of_gamma = 0;
    for (size_t i = 0; i < a->size; i++) {
        sum_of_gamma += gsl_sf_lngamma(a->data[i]);
    }

    // ((a-1) * logp).sum()
    double m1_logp = 0;
    for (size_t i = 0; i < a->size; i++) {
        m1_logp += (a->data[i] - 1) * logp->data[i];
    }
    gsl_vector_free(logp);

    // return N*(gammaln(a.sum()) - gammaln(a).sum() + ((a - 1)*logp).sum())
    return D->size1 * (gamma_sum - sum_of_gamma + m1_logp);

}


gsl_vector *dirichlet_mle(gsl_matrix *D) {
    /*
     * Iteratively computes the maximum likelihood Dirichlet
     * distribution given the observed data D.
     * i.e. the parameters a which maximize p(D | a)
     *
     * D: N x K matrix
     *    N: number of observations
     *    K: Number of parameters
     */
    /* Move to args */
    double tol = 1e-7;
    double lambda = 0.01;
    double maxiter = 1000;

    // logp = log(D).mean(axis=0)
    gsl_vector *logp = log_col_mean(D);

    // a0 = _init_a(D)
    /* gsl_vector *a0 = _init_a(D); */
    // Just always start search at 1
    gsl_vector *a0 = gsl_vector_alloc(D->size2);
    gsl_vector_set_all(a0, 1.0);

    //double last_ll = loglikelihood(D, a0);


    for (int iter = 0; iter < maxiter; iter++) {
        // a1 = _ipsi(psi(a0.sum()) + logp)
        gsl_vector *scaled_logp = gsl_vector_alloc(logp->size);
        gsl_vector_memcpy(scaled_logp, logp);
        gsl_vector_add_constant(scaled_logp, gsl_sf_psi(vec_sum(a0)));

//        gsl_vector *a1 = vec_elementwise_func(scaled_logp, _ipsi);

        /* Regularization */
        /* gsl_vector *a1_scaled = gsl_vector_alloc(a1->size); */
        /* gsl_vector_memcpy(a1_scaled, a1); */
        /* gsl_vector_scale(a1_scaled, lambda); */
        /* gsl_vector_sub(a1, a1_scaled); */

        /* gsl_vector_free(a1_scaled); */

        gsl_vector *a1 = vec_elementwise_func(scaled_logp , _ipsi);
        gsl_vector_free(scaled_logp);

        // if abs(loglikelihood(D, a1)-loglikelihood(D, a0)) < tol:
        //     return a1
        //a0 = a1
        gsl_vector_free(a0);
        //double this_ll = loglikelihood(D, a1);
       // if (fabs(this_ll - last_ll) < tol) {
        //    return a1;
        //}
        a0 = a1;
        //last_ll = this_ll;
    }
    return a0;
}

// smoothed mle
gsl_vector *dirichlet_mle_s(gsl_matrix *D, gsl_vector * w, double weight) {
    /*
     * Iteratively computes the maximum likelihood Dirichlet
     * distribution given the observed data D.
     * i.e. the parameters a which maximize p(D | a)
     *
     * D: N x K matrix
     *    N: number of observations
     *    K: Number of parameters
     */
    /* Move to args */
    double tol = 1e-7;
    double maxiter = 100;
    double lambda = 0.1;

    // logp = log(D).mean(axis=0)
    gsl_vector *logp = log_col_mean_s(D, w, weight);

    // a0 = _init_a(D)
    /* gsl_vector *a0 = _init_a_s(D, w, weight); */
    gsl_vector *a0 = gsl_vector_alloc(D->size2);
    gsl_vector_set_all(a0, 1.0);

    double last_ll = loglikelihood_s(D, a0, w, weight);

    for (int iter = 0; iter < maxiter; iter++) {
        // a1 = _ipsi(psi(a0.sum()) + logp)
        gsl_vector *scaled_logp = gsl_vector_alloc(logp->size);
        gsl_vector_memcpy(scaled_logp, logp);
        gsl_vector_add_constant(scaled_logp, gsl_sf_psi(vec_sum(a0)));

        gsl_vector *a1 = vec_elementwise_func(scaled_logp, _ipsi);

        gsl_vector_free(scaled_logp);

        // if abs(loglikelihood(D, a1)-loglikelihood(D, a0)) < tol:
        //     return a1
        // a0 = a1
        gsl_vector_free(a0);
        double this_ll = loglikelihood_s(D, a1, w, weight);
        if (fabs(this_ll - last_ll) < tol) {
            return a1;
        }
        a0 = a1;
        last_ll = this_ll;
    }
    return a0;
}

gsl_vector *dirichlet_mle_descent(gsl_matrix *D) {
    /*
     * Iteratively computes the maximum likelihood Dirichlet
     * distribution given the observed data D.
     * i.e. the parameters a which maximize p(D | a)
     *
     * D: N x K matrix
     *    N: number of observations
     *    K: Number of parameters
     */
    /* Move to args */
    double tol = 1e-7;
    double maxiter = 10000;
    int status;
    int opt_iter;
    double f = 0.0;



    // logp = log(D).mean(axis=0)
    gsl_vector *logp = log_col_mean(D);

    // a0 = _init_a(D)
    gsl_vector *a0 = gsl_vector_alloc(D->size2);
    gsl_vector_set_all(a0, 1);
    //_init_a(D);

    dr_parameter param;
    param.log_p = logp;
    param.PENALTY = 0;

    const gsl_multimin_fdfminimizer_type * T;
    gsl_multimin_fdfminimizer * s;
    gsl_vector * x;
    gsl_multimin_function_fdf opt_fun;
    opt_fun.f = &dr_f;
    opt_fun.df = &dr_df;
    opt_fun.fdf = &dr_fdf;
    opt_fun.n = D->size2;
    opt_fun.params = (void*)(&param);
    x = gsl_vector_alloc(D->size2);

    for (size_t k = 0; k < D->size2; k++)
        gsl_vector_set(x, k, a0->data[k]);

    T = gsl_multimin_fdfminimizer_vector_bfgs;
    s = gsl_multimin_fdfminimizer_alloc(T, D->size2);
    gsl_multimin_fdfminimizer_set(s, &opt_fun, x, .01, 1e-4);

    opt_iter = 0;
    do
    {
        opt_iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);
       
        if (status)
        {
            printf("step: %02d -> f: %f\n", opt_iter-1, f);
            break;
        }
        
        status = gsl_multimin_test_gradient(s->gradient, 1e-3);
        //for (int k = 0; k < D->size2; k++)

       
        if (status == GSL_SUCCESS)
        {
           
            printf("succes step: %02d -> f: %f\n", opt_iter-1, f);
            break;

        }
        f = -s->f;
        //if ((opt_iter-1) % 10 == 0)
        printf("step: %02d -> f: %f\n", opt_iter-1, f);

    } while (status == GSL_CONTINUE && opt_iter < 4000);

    return x;
}


