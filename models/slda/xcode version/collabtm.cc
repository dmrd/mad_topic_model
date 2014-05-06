#include "collabtm.hh"

//#define INIT_TEST 1

CollabTM::CollabTM(Env &env, Ratings &ratings)
: _env(env), _ratings(ratings),
_nusers(env.nusers),
_ndocs(env.ndocs),
_nvocab(env.nvocab),
_k(env.k),
_iter(0),
_start_time(time(0)),

#ifdef INIT_TEST
_theta("theta", 0.3, (double)1000./_nusers, _ndocs,_k,&_r),
_beta("beta", 0.3, (double)1000./_ndocs, _nvocab,_k,&_r),
_x("x", 0.3, (double)1000./_ndocs, _nusers,_k,&_r),
_epsilon("epsilon", 0.3, (double)1000./_nusers, _ndocs,_k,&_r),
_a("a", 0.3, 0.3, _ndocs, &_r)
#else
_theta("theta", (double)1./_k, (double)1./_k, _ndocs,_k,&_r),
_beta("beta", (double)1./_k, (double)1./_k, _nvocab,_k,&_r),
_x("x", (double)1./_k, (double)1./_k, _nusers,_k,&_r),
_epsilon("epsilon", (double)1./_k, (double)1./_k, _ndocs,_k,&_r),
_a("a", (double)1./_k, (double)1./_k, _ndocs, &_r)
#endif
{
    gsl_rng_env_setup();
    const gsl_rng_type *T = gsl_rng_default;
    _r = gsl_rng_alloc(T);
    if (_env.seed)
        gsl_rng_set(_r, _env.seed);
    
    _af = fopen(Env::file_str("/logl.txt").c_str(), "w");
    if (!_af)  {
        printf("cannot open logl file:%s\n",  strerror(errno));
        exit(-1);
    }
}

void
CollabTM::initialize()
{
    if (_env.lda) {
        
        _beta.load_from_lda(_env.datfname, 0.01, _k); // eek! fixme.
        _theta.load_from_lda(_env.datfname, 0.1, _k);
        lerr("loaded lda fits");
        
    } else if (_env.use_docs) {
        _beta.initialize();
        _theta.initialize();
        
        _theta.initialize_exp();
        _beta.initialize_exp();
    }
    
    if (_env.use_ratings) {
        _x.initialize();
        _epsilon.initialize();
        _x.initialize_exp();
        _epsilon.initialize_exp();
        
        if (!_env.fixeda) {
            _a.initialize();
            _a.compute_expectations();
        }
    }
}


void
CollabTM::initialize_perturb_betas()
{
    if (_env.use_docs) {
        _beta.initialize();
        _theta.set_to_prior_curr();
        
        _theta.compute_expectations();
        _beta.compute_expectations();
    }
    
    if (_env.use_ratings) {
        _x.set_to_prior_curr();
        _epsilon.set_to_prior_curr();
        _x.compute_expectations();
        _epsilon.compute_expectations();
        
        if (!_env.fixeda) {
            _a.set_to_prior_curr();
            _a.compute_expectations();
        }
    }
}

void
CollabTM::get_phi(GPBase<Matrix> &a, uint32_t ai,
                  GPBase<Matrix> &b, uint32_t bi,
                  Array &phi)
{
    assert (phi.size() == a.k() &&
            phi.size() == b.k());
    assert (ai < a.n() && bi < b.n());
    const double  **eloga = a.expected_logv().const_data();
    const double  **elogb = b.expected_logv().const_data();
    phi.zero();
    for (uint32_t k = 0; k < _k; ++k)
        phi[k] = eloga[ai][k] + elogb[bi][k];
    phi.lognormalize();
}


void
CollabTM::get_xi(uint32_t nu, uint32_t nd,
                 Array &xi,
                 Array &xi_a,
                 Array &xi_b)
{
    assert (xi.size() == 2 *_k && xi_a.size() == _k && xi_b.size() == _k);
    const double  **elogx = _x.expected_logv().const_data();
    const double  **elogtheta = _theta.expected_logv().const_data();
    const double  **elogepsilon = _epsilon.expected_logv().const_data();
    xi.zero();
    for (uint32_t k = 0; k < 2*_k; ++k) {
        if (k < _k)
            xi[k] = elogx[nu][k] + elogtheta[nd][k];
        else {
            uint32_t t = k - _k;
            xi[k] = elogx[nu][t] + elogepsilon[nd][t];
        }
    }
    xi.lognormalize();
    for (uint32_t k = 0; k < 2*_k; ++k)
        if (k < _k)
            xi_a[k] = xi[k];
        else
            xi_b[k-_k] = xi[k];
}

void
CollabTM::batch_infer()
{
    if (_env.perturb_only_beta_shape)
        initialize_perturb_betas();
    else
        initialize();
    
    approx_log_likelihood();
    
    Array phi(_k);
    Array xi(2*_k);
    Array xi_a(_k);
    Array xi_b(_k);
    
    while(1) {
        
        if (_env.use_docs && !_env.lda) {
            
            for (uint32_t nd = 0; nd < _ndocs; ++nd) {
                
                const WordVec *w = _ratings.get_words(nd);
                for (uint32_t nw = 0; w && nw < w->size(); nw++) {
                    WordCount p = (*w)[nw];
                    uint32_t word = p.first;
                    uint32_t count = p.second;
                    
                    get_phi(_theta, nd, _beta, word, phi);
                    if (count > 1)
                        phi.scale(count);
                    
                    _theta.update_shape_next(nd, phi);
                    _beta.update_shape_next(word, phi);
                }
            }
        }
        
        if (_env.use_ratings) {
            for (uint32_t nu = 0; nu < _nusers; ++nu) {
                
                const vector<uint32_t> *docs = _ratings.get_movies(nu);
                for (uint32_t j = 0; j < docs->size(); ++j) {
                    uint32_t nd = (*docs)[j];
                    yval_t y = _ratings.r(nu,nd);
                    
                    assert (y > 0);
                    
                    if (_env.use_docs) {
                        get_xi(nu, nd, xi, xi_a, xi_b);
                        if (y > 1) {
                            xi_a.scale(y);
                            xi_b.scale(y);
                            xi.scale(y);
                        }
                        if (!_env.lda)
                            _theta.update_shape_next(nd, xi_a);
                        _epsilon.update_shape_next(nd, xi_b);
                        _x.update_shape_next(nu, xi_a);
                        _x.update_shape_next(nu, xi_b);
                        
                        if (!_env.fixeda)
                            _a.update_shape_next(nd, y); // since \sum_k \xi_k = 1
                        
                    } else { // ratings only
                        
                        get_phi(_x, nu, _epsilon, nd, phi);
                        if (y > 1)
                            phi.scale(y);
                        
                        _x.update_shape_next(nu, phi);
                        _epsilon.update_shape_next(nd, phi);
                    }
                }
            }
        }
        
        if (_env.vb || (_env.vbinit && _iter < _env.vbinit_iter))
            update_all_rates_in_seq();
        else {
            update_all_rates();
            swap_all();
            compute_all_expectations();
        }
        
        if (_iter % 10 == 0) {
            lerr("Iteration %d\n", _iter);
            approx_log_likelihood();
            save_model();
        }
        if (_env.save_state_now)
            exit(0);
        
        _iter++;
    }
}


void
CollabTM::update_all_rates()
{
    if (_env.use_docs && !_env.lda) {
        // update theta rate
        Array betasum(_k);
        _beta.sum_rows(betasum);
        _theta.update_rate_next(betasum);
        
        // update beta rate
        Array thetasum(_k);
        _theta.sum_rows(thetasum);
        _beta.update_rate_next(thetasum);
    }
    
    if (_env.use_ratings) {
        // update theta rate
        Array xsum(_k);
        _x.sum_rows(xsum);
        
        if (!_env.lda)
            if (!_env.fixeda)
                _theta.update_rate_next(xsum, _a.expected_v());
            else
                _theta.update_rate_next(xsum);
        
        // update x rate
        if (_env.use_docs) {
            Array scaledthetasum(_k);
            if (!_env.fixeda)
                _theta.scaled_sum_rows(scaledthetasum, _a.expected_v());
            else
                _theta.sum_rows(scaledthetasum);
            _x.update_rate_next(scaledthetasum);
        }
        
        Array scaledepsilonsum(_k);
        if (!_env.fixeda)
            _epsilon.scaled_sum_rows(scaledepsilonsum, _a.expected_v());
        else
            _epsilon.sum_rows(scaledepsilonsum);
        _x.update_rate_next(scaledepsilonsum);
        
        // update epsilon rate
        if (_env.fixeda)
            _epsilon.update_rate_next(xsum);
        else
            _epsilon.update_rate_next(xsum, _a.expected_v());
        
        if (!_env.fixeda ) {
            // update 'a' rate
            Array arate(_ndocs);
            Matrix &theta_ev = _theta.expected_v();
            const double **theta_evd = theta_ev.const_data();
            Matrix &epsilon_ev = _epsilon.expected_v();
            const double **epsilon_evd = epsilon_ev.const_data();
            for (uint32_t nd = 0; nd < _ndocs; ++nd)
                for (uint32_t k = 0; k < _k; ++k)
                    arate[nd] += xsum[k] * (theta_evd[nd][k] + epsilon_evd[nd][k]);
            _a.update_rate_next(arate);
        }
    }
}



void
CollabTM::swap_all()
{
    if (_env.use_docs && !_env.lda) {
        _theta.swap();
        _beta.swap();
    }
    if (_env.use_ratings) {
        _epsilon.swap();
        if (!_env.fixeda)
            _a.swap();
        _x.swap();
    }
}

void
CollabTM::compute_all_expectations()
{
    if (_env.use_docs && !_env.lda) {
        _theta.compute_expectations();
        _beta.compute_expectations();
    }
    
    if (_env.use_ratings) {
        _epsilon.compute_expectations();
        if (!_env.fixeda)
            _a.compute_expectations();
        _x.compute_expectations();
    }
}

void
CollabTM::approx_log_likelihood()
{
    return; // XXX
    if (_nusers > 10000 || _k > 10)
        return;
    
    const double ** etheta = _theta.expected_v().const_data();
    const double ** elogtheta = _theta.expected_logv().const_data();
    const double ** ebeta = _beta.expected_v().const_data();
    const double ** elogbeta = _beta.expected_logv().const_data();
    const double ** ex = _x.expected_v().const_data();
    const double ** elogx = _x.expected_logv().const_data();
    const double ** eepsilon = _epsilon.expected_v().const_data();
    const double ** elogepsilon = _epsilon.expected_logv().const_data();
    
    const double *ea = _env.fixeda? NULL : _a.expected_v().const_data();
    const double *eloga = _env.fixeda ? NULL : _a.expected_logv().const_data();
    
    double s = .0;
    Array phi(_k);
    Array xi(2*_k);
    Array xi_a(_k);
    Array xi_b(_k);
    
    for (uint32_t nd = 0; nd < _ndocs; ++nd) {
        const WordVec *w = _ratings.get_words(nd);
        
        for (uint32_t nw = 0; w && nw < w->size(); nw++) {
            WordCount p = (*w)[nw];
            uint32_t word = p.first;
            uint32_t count = p.second;
            
            get_phi(_theta, nd, _beta, word, phi);
            
            double v = .0;
            for (uint32_t k = 0; k < _k; ++k) 
                v += count * phi[k] * (elogtheta[nd][k] +			\
                                       elogbeta[word][k] - log(phi[k]));
            s += v;
            
            for (uint32_t k = 0; k < _k; ++k)
                s -= etheta[nd][k] * ebeta[word][k];
        }
    }
    
    debug("E1: s = %f\n", s);
    
    for (uint32_t nu = 0; nu < _nusers; ++nu) {
        const vector<uint32_t> *docs = _ratings.get_movies(nu);
        
        for (uint32_t j = 0; j < docs->size(); ++j) {
            uint32_t nd = (*docs)[j];
            yval_t y = _ratings.r(nu,nd);
            
            assert (y > 0);
            
            get_xi(nu, nd, xi, xi_a, xi_b);
            
            debug("xi = %s\n", xi.s().c_str());
            
            double v = .0;
            for (uint32_t k = 0; k < 2*_k; ++k) {
                double r = .0;
                if (k < _k)
                    r = !_env.fixeda ? (elogx[nu][k] + elogtheta[nd][k] + eloga[nd]) : 
                    (elogx[nu][k] + elogtheta[nd][k]);
                else {
                    uint32_t t = k - _k;
                    r = !_env.fixeda ? (elogx[nu][t] + elogepsilon[nd][t] + eloga[nd]) :
                    (elogx[nu][t] + elogepsilon[nd][t]);
                }
                v += y * xi[k] * (r - log(xi[k]));
            }
            s += v;
            
            for (uint32_t k = 0; k < 2*_k; ++k) {
                double r = .0;
                if (k < _k)
                    r = !_env.fixeda ? (ex[nu][k] * etheta[nd][k] * ea[nd]) : \
                    (ex[nu][k] * etheta[nd][k]);
                else {
                    uint32_t t = k - _k;
                    r = !_env.fixeda ? (ex[nu][t] * eepsilon[nd][t] * ea[nd]) : \
                    (ex[nu][t] * eepsilon[nd][t]);
                }
                s -= r;
            }
        }
    }
    
    debug("E2: s = %f\n", s);
    
    s += _theta.compute_elbo_term();
    s += _beta.compute_elbo_term();
    s += _x.compute_elbo_term();
    s += _epsilon.compute_elbo_term();
    if (!_env.fixeda)
        s += _a.compute_elbo_term();
    
    debug("E3: s = %f\n", s);
    
    fprintf(_af, "%.5f\n", s);
    fflush(_af);
}

void
CollabTM::save_model()
{
    IDMap idmap; // null map
    if (_env.use_ratings) {
        printf("saving ratings state\n");
        fflush(stdout);
        _x.save_state(_ratings.seq2user());
        _epsilon.save_state(_ratings.seq2movie());
        if (!_env.fixeda) 
            _a.save_state(_ratings.seq2movie());
    }
    
    if (_env.use_docs) {
        _theta.save_state(_ratings.seq2movie());
        _beta.save_state(idmap);
    }
}
void
CollabTM::update_all_rates_in_seq()
{
    if (_env.use_docs && !_env.lda) {
        // update theta rate
        Array betasum(_k);
        _beta.sum_rows(betasum);
        _theta.update_rate_next(betasum);
    }
    
    Array xsum(_k);
    if (_env.use_ratings) {
        _x.sum_rows(xsum);
        if (!_env.lda)
            if (_env.fixeda)
                _theta.update_rate_next(xsum);
            else
                _theta.update_rate_next(xsum, _a.expected_v());
    }
    
    if (!_env.lda) {
        _theta.swap();
        _theta.compute_expectations();
    }
    
    if (_env.use_docs && !_env.lda) {
        // update beta rate
        Array thetasum(_k);
        _theta.sum_rows(thetasum);
        _beta.update_rate_next(thetasum);
        
        _beta.swap();
        _beta.compute_expectations();
    }
    
    if (_env.use_ratings) {
        // update x rate
        Array scaledthetasum(_k);
        if (!_env.fixeda)
            _theta.scaled_sum_rows(scaledthetasum, _a.expected_v());
        else
            _theta.sum_rows(scaledthetasum);
        
        Array scaledepsilonsum(_k);
        if (!_env.fixeda)
            _epsilon.scaled_sum_rows(scaledepsilonsum, _a.expected_v());
        else
            _epsilon.sum_rows(scaledepsilonsum);
        
        _x.update_rate_next(scaledthetasum);
        _x.update_rate_next(scaledepsilonsum);
        
        _x.swap();
        _x.compute_expectations();
        
        // update epsilon rate
        if (_env.fixeda)
            _epsilon.update_rate_next(xsum);
        else
            _epsilon.update_rate_next(xsum, _a.expected_v());
        
        _epsilon.swap();
        _epsilon.compute_expectations();
        
        if (!_env.fixeda) {
            // update 'a' rate
            Array arate(_ndocs);
            Matrix &theta_ev = _theta.expected_v();
            const double **theta_evd = theta_ev.const_data();
            Matrix &epsilon_ev = _epsilon.expected_v();
            const double **epsilon_evd = epsilon_ev.const_data();
            for (uint32_t nd = 0; nd < _ndocs; ++nd)
                for (uint32_t k = 0; k < _k; ++k)
                    arate[nd] += xsum[k] * (theta_evd[nd][k] + epsilon_evd[nd][k]);
            _a.update_rate_next(arate);
            _a.swap();
            _a.compute_expectations();
        }
    }
}


