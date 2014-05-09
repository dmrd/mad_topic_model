#ifndef DIRICHLET_H
#define DIRICHLET_H
#include <gsl/gsl_matrix.h>

gsl_vector *dirichlet_mle(gsl_matrix *D);
double _ipsi(double y);
gsl_vector *dirichlet_mle_s(gsl_matrix *D, gsl_vector * w, double weight);

#endif
