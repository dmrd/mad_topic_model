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
#ifndef OPT_H_INCLUDED
#define OPT_H_INCLUDED
#include <gsl/gsl_vector.h>
#include "slda.h"
#include <vector>
#include <gsl/gsl_rng.h>
using namespace std;

/*
 * structure for the gsl optimization routine
 *
 */

struct opt_parameter
{
	std::vector<suffstats *> ss;
	slda * model;
	double PENALTY;
};

/*
 * function to compute the value of the obj function, then 
 * return it
 */

double softmax_f(const gsl_vector * x, void * opt_param);

/*
 * function to compute the derivatives of function 
 *
 */

void softmax_df(const gsl_vector * x, void * opt_param, gsl_vector * df);

/*
 * function to compute the value and derivatives of the function 
 *
 */

void softmax_fdf(const gsl_vector * x, void * opt_param, double * f, gsl_vector * df);


/**
struct stoch_opt_parameter
{
	std::vector<suffstats *> ss;
	std::vector<int> stoch_authors;
	std::vector<int> stoch_docs;

	gsl_rng * rng;
	int author_trials ;
    int doc_trials;

	std::vector<double> author_prob;
	std::vector<double> doc_prob;

	slda * model;
	double PENALTY;
};

//function to compute softmax objective using stochastic sampling
double softmax_f_stoch(const gsl_vector *, void * opt_param, double * f, gsl_vector * df );
void softmax_df_stoch(const gsl_vector * x, void * opt_param, gsl_vector * df);
void softmax_fdf_stoch(const gsl_vector * x, void * opt_param, double * f, gsl_vector * df);
**/

#endif // OPT_H_INCLUDED

