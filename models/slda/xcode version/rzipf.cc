#include "collabtm.hh"

//#define INIT_TEST 1

CollabTM::CollabTM(Env &env, Ratings &ratings)
: _env(env), _ratings(ratings),
_nusers(env.nusers),
_ndocs(env.ndocs),
_nvocab(env.nvocab),
_k(env.k),
_k2(env.k2),
_l(env.l),
_iter(0),
_start_time(time(0)),
_chiSize(0); //REMEMBER TO SET THIS LATER

#ifdef INIT_TEST
// document params
_theta("theta", 0.3, (double)1000./_nusers, _ndocs,_k,&_r),
_beta("beta", 0.3, (double)1000./_ndocs, _nvocab,_k,&_r),
//latent preferences
_etaU("user eta", 0.3, (double)1000./_ndocs, _nusers,_k,&_r),
_etaI("item eta", 0.3, (double)1000./_nusers, _ndocs,_k,&_r),
//latent communities
_epsilonU("user epsilon", 0.3, (double)1000./_ndocs, _nusers,_k2,&_r),
_epsilonI("item epsilon", 0.3, (double)1000./_nusers, _ndocs,_k2,&_r),
//locations
_lambdaU("user lambda", 0.3, (double)1000./_ndocs, _nusers,_l,&_r),
_lambdaI("item lambda", 0.3, (double)1000./_nusers, _ndocs,_l,&_r)

// scalings
_a1("a1", 0.3, 0.3, _nusers, &_r)
_a1("a2", 0.3, 0.3, _ndocs, &_r)
_a3("a3", 0.3, 0.3, _ndocs, &_r)
_s("a3", 0.3, 0.3, _ndocs, &_r)


#else
// document params
_theta("theta", (double)1./_k, (double)1./_k, _ndocs,_k,&_r),
_beta("beta", (double)1./_k, (double)1./_k, _nvocab,_k,&_r),
// latent preferences
_etaU("user eta", (double)1./_k, (double)1./_k, _nusers,_k,&_r),
_etaI("item eta", (double)1./_k, (double)1./_k, _ndocs,_k,&_r),
//locations
_lambdaU("lambda eta", (double)1./_k, (double)1./_k, _nusers,_k,&_r),
_lambdaI("lambda eta", (double)1./_k, (double)1./_k, _ndocs,_k,&_r),
// latent communities
_epsilonU("user epsilon", (double)1./_k, (double)1./_k, _nusers,_k2,&_r),
_epsilonI("item epsilon", (double)1./_k, (double)1./_k, _ndocs,_k2,&_r),
//scalings
_a1("a1", (double)1./_k, (double)1./_k, _nusers, &_r)
_a2("a2", (double)1./_k, (double)1./_k, _ndocs, &_r)
_a3("a3", (double)1./_k, (double)1./_k, _ndocs, &_r)
_s("a3", (double)1./_k, (double)1./_k, _ndocs, &_r)


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
    
    _beta.initialize();
    _theta.initialize();
        
    _theta.initialize_exp();
    _beta.initialize_exp();
  
    _x.initialize();
    _epsilon.initialize();
    _x.initialize_exp();
    _epsilon.initialize_exp();
        
    if(!_env.fixeda) {
        _a1.initialize();
        _a1.compute_expectations();
        _a2.initialize();
        _a2.compute_expectations();

        
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
            _a1.set_to_prior_curr();
            _a1.compute_expectations();
            _a2.set_to_prior_curr();
            _a2.compute_expectations();
        }
    }
}

void
CollabTM::batch_infer()
{
    if (_env.perturb_only_beta_shape)
        initialize_perturb_betas();
    else
        initialize();
    
    approx_log_likelihood();
    

    while(1) {
        
        update_all_rates();
        swap_all();
        compute_all_expectations();
        
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

// location counts. NON TRIVIAL.
void
rzipf:get_gamma(GPBase<Matrix> &a, uint32_t ai,
               GPBase<Matrix> &b, uint32_t bi,
               Array &gamma)
{
    assert (zeta.size() == a.k() &&
            zeta.size() == b.k());
    assert (ai < a.n() && bi < b.n());
    const double  **eloga = a.expected_logv().const_data();
    const double  **elogb = b.expected_logv().const_data();
    phi.zero();
    for (uint32_t k = 0; k < _k; ++k)
        zeta[k] = eloga[ai][k] + elogb[bi][k];
    zeta.lognormalize();
}

// preference counts without text. DONE.
void
rzipf:get_xi_nodocs(GPBase<Matrix> &a, uint32_t ai,
                GPBase<Matrix> &b, uint32_t bi,
                Array &xi)
{
    assert (xi.size() == a.k() &&
            xi.size() == b.k());
    assert (ai < a.n() && bi < b.n());
    const double  **eloga = a.expected_logv().const_data();
    const double  **elogb = b.expected_logv().const_data();
    xi.zero();
    for (uint32_t k = 0; k < _k; ++k)
        xi[k] = eloga[ai][k] + elogb[bi][k];
    xi.lognormalize();
    return;
    
}

// preference counts with text. DONE
void
rzipf:get_xi_doc(uint32_t ai, uint32_t bi, Array &xi, Array &xi_a, Array &xi_b)
{
    assert (xi.size() == 2*_k &&
            xi.size() == 2*_k &&
            xi_a.size() == _k && xi_b.size() == _k );

    const double **elogetaU = _etaU.expected_logv().const_data();
    const double **elogetaI = _etaI.expected_logv().const_data();
    const double **elogbeta = _beta.expected_logv().const_data();
    const double elogs = _s.expected_logv().const_data()[bi];
    
    xi.zero();
    xi_a.zero();
    xi_b.zero();
    
    for (uint32_t k = 0; k < _k; ++k)
    {
        xi[k] = elogetaU[ai][k] + elogetaI[bi][k];
        xi[k+_k] = elogetaU[ai][k] + elogbeta[bi][k]+elogs;
    }
    
    xi.lognormalize();
    for (uint32_t k = 0; k <_k; ++k)
        xi_a[k] = xi[k];
    for (uint32_t k = _k; k <2*_k; ++k)
        xi_b[k] = xi[k];
    return;
}

//topic counts. DONE.
 void
 rzipf::get_phi(GPBase<Matrix> &a, uint32_t ai,
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

// observation counts. DONE.
void
zipf:get_chi(uint32_t ai, uint32_t bi,
           Array &chi, double loga1, double loga2)
{
    assert(chi.size() == _chiSize);
    chi.zero();
    uint32_t count = 0;
    
    const double  **elogetaU = _etaU.expected_logv().const_data();
    const double  **elogetaI = _etaI.expected_logv().const_data();
    
    for (uint32_t k = 0; k < _k; ++k)
        chi[k] = loga1 + loga2 + elogetaI[bi][k] + elogetaU[ai][k];
    count+=_k;
    
    if(env.use_docs)
    {
        const double  **elogtheta = _theta.expected_logv().const_data();
        const double  **elogbeta = _beta.expected_logv().const_data();
        const double elogs = _s.expected_logv().const_data()[bi];
        
        for (uint32_t k = 0; k < _k; ++k)
        {
            uint32_t kChi = count + k;
            chi[kChi] = loga1 + loga2 + elogs + elogbeta[bi][k] + elogetaU[ai][k];
        }
        
        count+=k;
        
    }
    if(env.use_locs)
    {
        const double **eloglambdaU = _lambdaU.expected_logv().const_data();
        const double **eloglambdaI = _lambdaI.expected_logv().const_data();
        const double loga3 = _a3.expected_logv().const_data()[bi];
        for (uint32_t k = 0; k < _l; ++k)
        {
            uint32_t kChi = count + k;
            chi[k] =  eloglambdaU[ai][k] + eloglambdaI[bi][k];
        }
        count+=_l;
    }
    
    if(env.use_latent)
    {
        const double **elogepsilonI = _epsilonI.expected_logv().const_data();
        const double **elogepsilonU = _epsilonU.expected_logv().const_data;
        for (uint32_t k = 0; k < _k2; ++k)
        {
            uint32_t kChi = count + k;
            chi[k] = elogepsilonU[ai][k] + elogepsilonI[bi][k];
        }
        count+=_k2;
    }
    
    xi.lognormalize()
}


// done
void
zipf::add_in_xi_nodoc()
{
    Array xi(_k);
    for (uint32_t nu = 0; nu < _nusers; ++n)
    {
        const vector<uint32_t> *docs = _ratings.get_movies(nu);
        for (uint32_t j = 0; j < docs->size(); ++j)
        {
            uint32_t nd = (*docs)[j];
            yval_t y = _ratings.r(nu,nd);
            assert(y>0);
            
            get_xi_nodocs(_etaU,nu,_etaI,nd,xi);
            if (y>1)
                xi.scale(y);
                           
            _etaU.update_rate_next(nu,xi)
            _etaI.update_rate_next(nd,xi);
        }
    }
   
}

//done
void
zipf::add_in_xi_doc()
{
    Array xi(2*_k);
    Array xi_a(_k);
    Array xi_b(_k);

    for (uint32_t nu = 0; nu < _nusers; ++n)
    {
        const vector<uint32_t> *docs = _ratings.get_movies(nu);
        for (uint32_t j = 0; j < docs->size(); ++j)
        {
            uint32_t nd = (*docs)[j];
            yval_t y = _ratings.r(nu,nd);
            assert(y>0);
            
            get_xi_docs(nu,nd,xi,xi_a,xi_b);
            if (y>1)
            {
                xi.scale(y);
                xi_a.scale(y);
                xi_b.scale(y);
            }
            
            _etaU.update_rate_next(nu,xi)
            _etaI.update_rate_next(nd,xi_a);
            _theta.update_rate_next(nd,xi_b);
            _s.update_rate_next(nd,xi_b);

        }
    }
    
}

// do later
void
zipf::add_in_gamma()
{
    
}

// do later
void
zipf::add_in_phi()
{
}


// done
void
zipf::add_in_chi()
{
    Array chi(_chiSize);
    const double **eloga1 = _a1.expected_logv.const_data();
    const double **eloga2 = _a2.expected_logv.const_data();

    for (uint32_t nu = 0; nu < _nusers; ++nu)
    {
        const vector<uint32_t> *docs = _ratings.get_movies(nu);
        for (uint32_t j = 0; j < docs->size(); ++j)
        {
            uint32_t nd = (*docs)[j];
            yval_t y = _ratings.r(nu,nd);
            assert (y > 0);
            
            get_chi(nu,nd,chi,eloga1[nu],eloga2[nd]);
            
            uint32_t count = 0;
            Array chi_a(_k) = subarr(chi,0,_k);
            _a1.update_shape_next(nu,chi_a);
            _a2.update_shape_next(nd,chi_a);
            _etaU.update_shape_next(nu,chi_a);
            _etaI.update_shape_next(nd,chi_a);
            count = count + _k;

            if (env.use_docs)
            {
                Array chi_b(_k);
                subarr(chi,count,chi_b);
                _a1.update_shape_next(nu,chi_b);
                _a2.update_shape_next(nd,chi_b);
                _etaU.update_shape_next(nu,chi_b);
                _beta.update_shape_next(nd,chi_a);
                _s.update_shape_next(nd,chi_a);
                count += _k;

            }
            if (env.use_locs)
            {
                Array chi_c(_l);
                subarr(chi,count,chi_c);
                _a3.update_shape_next(nd,chi_c);
                _lambdaI.update_shape_next(nd,chi_c);
                _lambdaU.update_shape_next(nu,chi_c);
                
                count+=_l;
            }
            if (env.use_latent)
            {
                Array chi_d(_k2);
                subarr(chi,count,_k2);
                _epsilonU.update_shape_next(nu,chi_d);
                _epsilonI.update_shape_next(nd,chi_d);
            }
            
        }
    }
}


void
CollabTM::update_all_shapes()
{
  
    
    if (env.use_docs)
    {
        add_in_xi_doc();
        add_in_phi();
    }
    else
        add_in_xi_nodoc();
    if (env.use_locs)
        add_in_gamma();
    
    add_in_chi();
    
}

void
CollabTM::update_all_rates()
{

    if (!_env.zipf)
    {
        // update theta rate
        Array betasum(_k);
        _beta.sum_rows(betasum);
        _theta.update_rate_next(betasum);
        
        // update beta rate
        Array thetasum(_k);
        _theta.sum_rows(thetasum);
        _beta.update_rate_next(thetasum);
        return;
    }
    
    //else, assume zero inflation
    Array scaledEtaUSum(_k);
    Array scaledEtaISum(_k);
    
    _etaU.scaled_sum_rows(scaledEtaUSum,_a1.expected_v);
    _etaI.scaled_sum_rows(scaledEtaISum,_a2.expected_v);
    _etaU.update_rate_next(scaledEtaISum,_a1.expected_v);
    _etaI.update_rate_next(scaledEtaUSum,_a2.expected_v);
    
    // latent rate updates
    if(env.use_latent)
    {
        Array epsilonUsum(_k2);
        _epsilonU.sum_rows(epsilonUsum);
        Array epsilonIsum(_k2);
        _epsilonI.sum_rows(epsilonIsum);
        
        _epsilonU.update_rate_next(epsilonIsum);
        _epsilonI.update_rate_next(epsilonUsum);
    }

    // rate updates for word counts
    Array scaledThetaSum(_k);
    if(_env.use_docs)
    {
        Array betasum(_k);
        Array thetasum(_k);
        
        _theta.sum_rows(thetasum);
        _beta.sum_rows(betasum);
        _theta.update_rate_next(betasum);
        _beta_update_rate_next(thetasum)
        
        
        Array a2sdot(_k);
        hadamard(a2.expected_v,_s.expected_v,a2sdot);
        
        _theta.scaled_sum_rows(scaledThetaSum, as2dot);
        _theta.update_rate_next(scaledEtaUSum, as2dot);
        _etaU.update_rate_next(scaledThetaSum, a1.expected_v);

    }
    
    //location rate updates
    if(_env.use_locs)
    {
        Array lambdaUsum(_l);
        Array lambdaIscale(_l);
        
        _lambdaU.sum_rows(lambdaUsum);
        _lambdaI.scaled_sum_rows(lambdaIscale, _a3.expected_v);
        
        _lambdaU.update_rate_next(lambdaIscale);
        _lambdaI.update_rate_next(lambdaUsum,_a3.expected_v);
    }
    
    sparseSum();
    scale_preferences(scaledEtaUSum, scaledEtaISum,scaledThetaSum);
    if (env.use_locs)
        scale_locs();
    
}
void
zipf::scale_preferences(Array &scaledEtaU, Array &scaledEtaI, Array &scaledTheta)
{
    // convert to points
    const double *scaledEtaSumU = scaledEtaU.const_data()
    const double *scaledEtaSumI = scaledEtaI.const_data();
    const double *scaledThetaSum = scaledTheta.const_data();

    //set up rate updates
    Array a1rate(_nusers);
    Array a2rate(_ndocs);
    Array srate(_ndocs);
    
    //get pointers to individual matrix elements
    Matrix &theta_ev = _theta.expected_v();
    const double **theta_evd = theta_ev.const_data();
    Matrix &etaU_ev = _etaU.expected_v();
    const double **etaU_evd = etaU_ev.const_data();
    Matrix &etaI_ev = _etaI.expected_v();
    const double **etaI_evd = etaI_ev.const_data();
    
    //get pointers to scaling terms
    Array &a2_ev = _a2.expected_v();
    Array &s_ev = _s.expected_v();
    const double *a2_evd = a2_ev.const_data();
    const double *s_evd = s_evd.const_data();
    
    

    for (uint32_t k = 0; k < _k; ++k)
    {
        //get scaled sum for a given parameter index
        double etaUv = scaledEtaSumU[k];
        double etaIv = scaledEtaSumI[k];
        double thetaBoth = scaledThetaSum[k];
            
        for (uint32_t nu = 0; nu<_nusers: ++nu)
        {
            a1rate[nu]+=etaIv*etaU[nu][k];
            if (env.use_docs)
                a1rate[nu]+=thetaBoth*etaU[nu][k];
        }
        for (uint32_t nd = 0; nd < _ndocs; ++nd)
        {
            a2rate[nd]+= etaUv*etaIv[nd][k];
            if(env.use_docs)
            {
                a2rate[nd]+=sV[nd]*theta[nd][k]*etaUv;
                s_rate[nd]+=a2V[nd]*theta[nd][k]*etaUv;
            }
    
        }
        
    }
    _a1.update_rate_next(a1rate);
    _a2.update_rate_next(a2rate);
    if(env.use_docs)
        _s.update_rate_next(srate);
    
}

//compute scales for locations
void
zipf::scale_locs(Array &lambdaUsum)
{
    Array a3rate[_ndocs];
    const double *lamU = lambdaUsum.const_data();
    Matrix &lambdaI = _lambdaI.expected_v();
    const double **lamI = lambdaI.const_data();
    
    for (uint32_t nd = 0; nd < _ndocs; ++nd )
        for (uint32_t l=0; l < _l; ++l)
            a3rate[nd]+=lamI[nd][l]*lamU[l];
    _a3.update_rate_next(a3rate);
}

//sum up all zero inflated terms
void
rzipf::sparseSum()
{
    Matrix &etaU_ev = _etaU.expected_v();
    Matrix &etaI_ev = _etaI.expected_v();
    Matrix &theta_ev = _theta.expected_v();
    
    const double **etaU_evd = etaU_ev.const_data();
    const double **etaI_evd = etaI_ev.const_data();
    const double **theta_evd = theta_ev.const_data();
    
    Array &s_ev = _s.expected_v();
    const double *s_evd = s_ev.const_data();
    
    Array[_ndocs] srate;
    srate.zero();

    Array etaUSlice(_k);
    Array etaISlice(_k);
    Array thetaSlice(_k);
    Array thetaSlice2(_k);
    
    for (uint32_t nu = 0; nu < _nusers; ++nu)
    {
        const vector<uint32_t> *docs = _ratings.get_movies(nu);
        for (uint32_t j = 0; j < docs->size(); ++j)
        {
            uint32_t nd = (*docs)[j];
            
            etaU_ev.slice(0,nu,etaUSlice);
            etaI_ev.slice(0,nd,etaISlice);
            
            _etaU.update_rate_next(nu,etaISlice);
            _etaI.update_rate_next(nd,etaUSlice);
            if (env.use_docs)
            {
                theta_ev.slice(0,nd,thetaSlice);
                hadamard(s_ev,thetaAr,thetaSlice2);
                _etaU.update_rate_next(nu,thetaSlice2);
                
                etaUSlice.scale(s_evd[nd]);
                _theta.update_rate_next(nd,etaUSlice);
                for (uint32_t k = 0; k < _k; ++k)
                {
                    srate[nd]+=theta_evd[nd][k]*etaU_evd[nu][k];
                }
            }
        }
    }
    _s.update_rate_next(srate);
}

void
rzipf::swap_all()
{
    
    _theta.swap();
    _beta.swap();
    
    if (_env.zipf) {
        _epsilon.swap();
        if (!_env.fixeda)
        {
            _a1.swap();
            _a2.swap();
        }
        _x.swap();
    }
}

void
rzipf::compute_all_expectations()
{
    
    _theta.compute_expectations();
    _beta.compute_expectations();
    
    if (_env.zipf) {
        _epsilon.compute_expectations();
        if (!_env.fixeda)
            _a1.compute_expectations();
            _a2.compute_expectations();
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

/**
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
 **/





/**void
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
 
 **/


