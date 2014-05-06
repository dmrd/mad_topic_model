#include "ZIPF.hh"

ZIPF::ZIPF(Env &env, Ratings &ratings)
  : _env(env), _ratings(ratings),
    _n(env.n), _m(env.m), _k(env.k),_k2(4),
    _iter(0), _epsilon(0),
    _a(env.a), _b(env.b), // priors on theta
    _c(env.c), _d(env.d), // briors on beta
    _start_time(time(0)),
    _obsPosterior(0),
    _ratingPost(0),
    batch_infer1(false),


    _maxUserDeg(5), _maxItemDeg(5),
    _userDegree(_n), _itemDegree(_m),

    _acurr(_n,_k), _bcurr(_n,_k),
    _ccurr(_m,_k), _dcurr(_m,_k),
    _anext(_n,_k), _bnext(_n,_k),
    _cnext(_m,_k), _dnext(_m,_k),
    _s_epsilon(1),
    _bcurr2(_k),
    _dcurr2(_k),
    _bnext2(_k),
    _dnext2(_k),

    elogPBigPhi(0), elogQBigPhi(0),
   sEntropy(0),
   _pLikelihood(0), _qLikelihood(0),
   _miscount(0),

    _Elogtheta(_n,_k), _Etheta(_n,_k),
    _Elogbeta(_m,_k), _Ebeta(_m,_k),
    _etheta_sum(_m,_k), _ebeta_sum(_n,_k),
  _etheta_sum2(_k), _ebeta_sum2(_k),
    _etheta_sumOld(_m,_k), _ebeta_sumOld(_n,_k),




    _alpha(_k2), _alpha0(0), _psi(_k2), // community strength parameters
    _piU(_n,_k2), _piI(_n,_k2), _xiU(_n), _xiI(_m), // comunity belonging  and node popularities
    _mu0(0.0), _sigma0(1.0), // variances and std devs
    _mu1I(0.0), _mu1U(0.0),
     _sigma1I(10.0),_sigma1U(10.0),




//
//    _gammaU(_n,_k2), _lambdaU(_n),
//    _gammaUNew(_n,_k2), _lambdaUNew(_n),//user Gamma and Lambda
//    _gammaI(_m,_k2),  _lambdaI(_m),
//    _gammaINew(_m,_k2),  _lambdaINew(_m),
//
//    _sigma_xiU(10), _sigma_xiI(10), //
//    _sigma_xiUt(10), _sigma_xiIt(10), //
//    _sigma_psi(1), _sigma_psit(1),
//
//    _sigma_xiU2(.01), _sigma_xiI2(.01), //
//     _mu(_k2), _muNew(_k2),
//     _s(_n,_m), _sNew(_n,_m),
//    _l(_n,_m),_lNew(_n,_m),
//    _t(_n,_m),_tNew(_n,_m),
//    _bigPhi(_n,_m,_k2), _bigPhiNew(_n,_m,_k2),
//
//    _eExpPsi(_k2),_eExpXiU(_n), _eExpXiI(_m),
//    _eLogPiU(_n,_k2), _eLogPiI(_m,_k2),
//    _eLogPiUNew(_n,_k2), _eLogPiINew(_m,_k2),

    _tau0(65536),_muTau0(65536*2), //downweighting
    _kappa(0.5),_muKappa(0.9), // learning rate
    _rho(0.001),_muRho(0.001),
    _ratingELBO(0), _ratingLikelihood(0)

{
   
    
    gsl_rng_env_setup();
    const gsl_rng_type *T = gsl_rng_default;
    _r = gsl_rng_alloc(T);
    
   
    
    /**
    _hf = fopen(Env::file_str("/heldout.txt").c_str(), "w");
    if (!_hf)  {
        printf("cannot open heldout file:%s\n",  strerror(errno));
        exit(-1);
    }
    _af = fopen(Env::file_str("/logl.txt").c_str(), "w");
    if (!_af)  {
        printf("cannot open logl file:%s\n",  strerror(errno));
        exit(-1);
    }
    **/
}

ZIPF::~ZIPF()
{
    fclose(_hf);
    fclose(_vf);
    fclose(_af);
}

// Note: test set is loaded only to compute precision
void
ZIPF::load_validation_and_test_sets()
{
    char buf[4096];
    sprintf(buf, "%s/validation.tsv", _env.datfname.c_str());
    FILE *validf = fopen(buf, "r");
    assert(validf);
    _ratings.read_generic(validf, &_validation_map);
    fclose(validf);
    /**
    sprintf(buf, "%s/test.tsv", _env.datfname.c_str());
    FILE *testf = fopen(buf, "r");
    assert(testf);
    _ratings.read_generic(testf, &_test_map);
    fclose(testf);
    printf("+ loaded validation and test sets from %s\n", _env.datfname.c_str());
    fflush(stdout);
    Env::plog("test ratings", _test_map.size());
    Env::plog("validation ratings", _validation_map.size());
     **/
}

int
ZIPF::load_gamma()
{
    printf("loading gamma \n");
    fflush(stdout);
    
    fprintf(stderr, "+ loading gamma\n");
    double **gdU = _gammaU.data();
    double **gdI = _gammaI.data();
    FILE *gammaf = fopen("gamma.txt", "r");
    if (!gammaf)
        return -1;
    
    uint32_t n = 0;
    int sz = 32*_k;
    char *line = (char *)malloc(sz);
    
    while (!feof(gammaf)) {
        if (fgets(line, sz, gammaf) == NULL)
            break;
        //assert (fscanf(gammaf, "%[^\n]", line) > 0);
        debug("line = %s\n", line);
        uint32_t k = 0;
        char *p = line;
        do {
            char *q = NULL;
            double d = strtod(p, &q);
            if (q == p) {
                if (k < _k - 1) {
                    fprintf(stderr, "error parsing gamma file\n");
                    assert(0);
                }
                break;
            }
            p = q;
            if (k >= 2) // skip node id and seq
            {
                if (n < _n)
                    gdU[n][k-2] = d;
                else
                    gdI[n-_n][k-2]=d;
            }
            
            k++;
        } while (p != NULL);
        n++;
        debug("read %d lines\n", n);
        memset(line, 0, sz);
    }
    assert (n == _n+_m);
    
    fclose(gammaf);
    //load_validation_and_test_sets();
    Env::plog("model load", true);
    return 0;
}


void
ZIPF::getDegrees()
{
    _maxUserDeg = 0;
    for (uint32_t n = 0; n < _n; ++n)
    {
        _userDegree[n] = _ratings.get_movies(n)->size();
        if (_userDegree[n]>_maxUserDeg)
            _maxUserDeg = _userDegree[n];
        if (_userDegree[n]==0)
        {
            printf("zero degree %d\n",n);
             fflush(stdout);
        }
    }
    
    _maxItemDeg = 0;
    for (uint32_t m = 0; m < _m; ++m)
    {
        _itemDegree[m] = _ratings.get_users(m)->size();
        if (_itemDegree[m]>_maxItemDeg)
            _maxItemDeg = _itemDegree[m];
    }
    printf("max degrees %d,%d\n",_maxUserDeg,_maxItemDeg);
    fflush(stdout);
    
}
void
    ZIPF::communityInitialize()
{
  //load_gamma();
  double **dU = _gammaU.data();
  double **dI = _gammaI.data();

  for (uint32_t n = 0; n < _n; ++n)
    for (uint32_t k = 0; k < _k2; ++k)  {
      double previous = dU[n][k];
      dU[n][k] = gsl_ran_gamma(_r, 100 * 1, 0.01);
    }
  for (uint32_t m = 0; m < _m; ++m)
    for (uint32_t k = 0; k < _k2; ++k)  {
      double previous = dI[m][k];
      dI[m][k] = gsl_ran_gamma(_r, 100 * 1, 0.01);
    }
   getDegrees();
  double *dUL = _lambdaU.data();
  double *dIL = _lambdaI.data();
  for (uint32_t n=0; n < _n; ++n)
  {
      //printf(" degree %d\n",_userDegree[n]/_maxUserDeg);
      //fflush(stdout);
      dUL[n] = gsl_sf_log(((double)_userDegree[n])/((double)_maxUserDeg))+
      (gsl_rng_uniform(_r)-1/2)*.01;
  }
  for (uint32_t m=0; m < _m; ++m)
  {
    dIL[m] = gsl_sf_log( ((double)_itemDegree[m])/((double)_maxItemDeg))
      +(gsl_rng_uniform(_r)-1/2)*.01;
  }
    init_bigPhi();
    setExpXiandPsi();
    set_t();
    set_l();
    _t.copy_from(_tNew);
    _l.copy_from(_lNew);
    _mu.zero();
    _alpha.set_elements(_env.alpha);
    _alpha0 = _alpha.sum();
    
}

// initializes interaction parameter
// as random draws from a dirichlet distribution
void
ZIPF::init_bigPhi()
{
    double*** phid = _bigPhi.data();
    double** gUd = _gammaU.data();
    double** gId = _gammaI.data();

    double alphaU[_k2];
    double alphaI[_k2];
    double thetaU[_k2];
    double thetaI[_k2];
    
    for (uint32_t n = 0; n < _n; ++n)
    {
        for (uint32_t k = 0; k < _k2; ++ k)
        {
            
            alphaU[k]=gUd[n][k];
            double p = alphaU[k];
            double p2 = 0;
        }
        
        for (uint32_t m = 0; m < _m; ++m)
        {
            for (uint32_t k = 0; k < _k2; ++ k)
                alphaI[k]=gId[m][k];
            
            gsl_ran_dirichlet(_r, _k2, alphaU, thetaU);
            gsl_ran_dirichlet(_r, _k2, alphaI, thetaI);
            
            for (uint32_t kk = 0; kk < _k2; ++kk)
            {
                phid[n][m][kk] = thetaU[kk]*thetaI[kk];
               double  p = phid[n][m][kk];
               double p3 = thetaU[kk];
                double p2 = 0;
                
            }
        }
    }
}

void
    ZIPF::init_s()
{
  for (uint32_t n = 0; n < _n; ++n)
  {
    double **sd = _s.data();
      double **sdNew = _sNew.data();
    for (uint32_t m = 0; m < _m; ++m) 
    {
        if (_ratings.r(n,m)>0)
        {
          sd[n][m]=1;
            sdNew[n][m]=1;
        
        }
        else
        {
            sd[n][m] = _s_epsilon;
      }
    }
  }
}
    

void
ZIPF::ratingInitalize()
{
  double **ad = _acurr.data();
  double **bd = _bcurr.data();
  double **cd = _ccurr.data();
  double **dd = _dcurr.data();
    double *bd2 = _bcurr2.data();
    double *dd2 = _dcurr2.data();
    
    for (uint32_t k = 0; k < _k; ++k)
    {
        bd2[k] = _b + 0.1 * gsl_rng_uniform(_r);
        dd2[k] = _d + 0.1 * gsl_rng_uniform(_r);
    }
    
    for (uint32_t k = 0; k < _k; ++k)
    {
        assert (bd2[k]);
        assert (dd2[k]);
    }
    
    
    
  for (uint32_t i = 0; i < _n; ++i)
  {
    for (uint32_t k = 0; k < _k; ++k)
    {
      ad[i][k] = _a + 0.01 * gsl_rng_uniform(_r);
      bd[i][k] = _b + 0.1 * gsl_rng_uniform(_r);
    }
  }
  
  for (uint32_t i = 0; i < _m; ++i)
  {
    for (uint32_t k = 0; k < _k; ++k)
    {
      cd[i][k] = _c + 0.01 * gsl_rng_uniform(_r);
      dd[i][k] = _d + 0.1 * gsl_rng_uniform(_r);
    }
  }
    
  set_gamma_exp_init(_acurr, _Etheta, _Elogtheta, _b);
  set_gamma_exp_init(_ccurr, _Ebeta, _Elogbeta, _d);
  set_etheta_sum();
  set_ebeta_sum();
    set_etheta_sum2();
    set_ebeta_sum2();
  set_to_prior_users(_anext, _bnext);
  set_to_prior_movies(_cnext, _dnext);
}

void
ZIPF::initialize()
{
    ratingInitalize();
    communityInitialize();
    init_s();
    //set_s();
}

void
ZIPF::update_rho()
{
    return;
}

void
ZIPF::update_global_state()
{
 
  setExpXiandPsi();

  // updates community membership parameters in parallel
  
  set_l();
  set_t();
  set_s();
    
    _l.swap(_lNew);
    _s.swap(_sNew);
    _t.swap(_tNew);
    
  setGamma();
  set_BigPhi();


  update_rho();
  setLambda();
 set_mu();
    
    set_gamma_exp(_acurr, _bcurr, _Etheta, _Elogtheta);
    set_gamma_exp(_ccurr, _dcurr, _Ebeta, _Elogbeta);
    
    set_etheta_sum();
    set_ebeta_sum();

  swapCommunityVar();

  set_to_prior_users(_anext, _bnext);
  set_to_prior_movies(_cnext, _dnext);
}


void
ZIPF::set_etheta_sum()
{
  _etheta_sum.zero();
    const double **etheta  = _Etheta.const_data();
    const double **sd = _s.const_data();
    double **ethetasum = _etheta_sum.data();
    for (uint32_t m = 0; m < _m; ++m)
    {
        for (uint32_t k = 0; k < _k; ++k) {
            for (uint32_t n = 0; n < _n; ++n)
            {
                ethetasum[m][k] += etheta[n][k]*sd[n][m];
            }
        }
  debug("etheta sum set to %s\n", _etheta_sum.s().c_str());
    }
}

void
ZIPF::set_ebeta_sum()
{
  _ebeta_sum.zero();
  const double **sd = _s.const_data();
  const double **ebeta  = _Ebeta.const_data();
  double **ebetasum = _ebeta_sum.data();

    for (uint32_t n = 0; n < _n; ++n)
    {
        
        for (uint32_t k = 0; k < _k; ++k) {
            for (uint32_t m = 0; m < _m; ++m)
                ebetasum[n][k] += ebeta[m][k]*sd[n][m];
        }
    }
  debug("ebeta sum set to %s\n", _ebeta_sum.s().c_str());
}

void 
ZIPF::set_mu()
{
    double const **sd = _s.const_data();
    double const **ld = _l.const_data();
    double const ***phiD = _bigPhi.const_data();
    double maxTemp = 0;
    double maxGrad = 0;
    double maxPhi = 0;
    double maxl = 0;
    
  for (uint32_t kk = 0; kk < _k2; ++kk)
  {
    double grad = (_mu0-_mu[kk])/(SQR(_sigma0));
      double m = _mu[kk];
    double temp = exp(_mu[kk]+SQR(_sigma_psi)/2);
      
    for (uint32_t n = 0; n < _n; ++n)
    {
      for (uint32_t m = 0; m < _m; ++m)
      {
          
          grad += phiD[n][m][kk]*(sd[n][m]-ld[n][m]*temp);
          if (ld[n][m]>maxl)
              maxl = ld[n][m];
          if (phiD[n][m][kk]>maxPhi)maxPhi = phiD[n][m][kk];
      }
    }
       m = _mu[kk];
      if (maxTemp < temp)
          maxTemp = temp;
      if (maxGrad < grad)
          maxGrad = grad;
    _muNew[kk] = m+ grad*_muRho;
  }
   printf("MaxTemp,maxGrad,maxPhi,maxl,%e,%e,%e,%e\n",
          maxTemp,maxGrad,maxPhi,maxl);
}


void
ZIPF::set_to_prior_users(Matrix &a, Matrix &b)
{
  a.set_elements(_a);
  b.set_elements(_b);
}

void
ZIPF::set_to_prior_movies(Matrix &c, Matrix &d)
{
  c.set_elements(_c);
  d.set_elements(_d);
}

void
ZIPF::set_to_prior_users2(Matrix &a, Array &b)
{
    a.set_elements(_a);
    b.set_elements(_b);
}

void
ZIPF::set_to_prior_movies2(Matrix &c, Array &d)
{
    c.set_elements(_c);
    d.set_elements(_d);
}

void
ZIPF::set_gamma_exp(const Matrix &a, const Matrix &b, Matrix &v1, Matrix &v2)
{
  const double ** const ad = a.const_data();
  const double ** const bd = b.const_data();
  //const double ** const sd = _s.const_data();
  double **vd1 = v1.data();
  double **vd2 = v2.data();
  for (uint32_t i = 0; i < a.m(); ++i)
    for (uint32_t j = 0; j < b.n(); ++j) {
      assert(bd[j]);
      vd1[i][j] = ad[i][j] / bd[i][j];
      vd2[i][j] = gsl_sf_psi(ad[i][j]) - log(bd[i][j]);
    }
}



void
ZIPF::set_gamma_exp_init(const Matrix &a, Matrix &v1, Matrix &v2, double v)
{
  const double ** const ad = a.const_data();
  double **vd1 = v1.data();
  double **vd2 = v2.data();
  
  Array b(_k);
  for (uint32_t i = 0; i < a.m(); ++i)
    for (uint32_t j = 0; j < b.n(); ++j) {
      b[j] = v + 0.1 * gsl_rng_uniform(_r);
      assert(b[j]);
      vd1[i][j] = ad[i][j] / b[j];
      vd2[i][j] = gsl_sf_psi(ad[i][j]) - log(b[j]);
    }
}



void
ZIPF::batch_infer()
{
    batch_infer1 = true;
    FILE *tf = fopen("/Users/msimchowitz/Documents/JP/elbo1.tsv", "w");
    assert(tf);
    
    printf("initializing \n");
    fflush(stdout);
    initialize();
    printf("done initializing \n");
    fflush(stdout);
    info("ZIPF initialization done\n");
  
  //approx_log_likelihood();
  fflush(stdout);

  Array phi(_k);

  while (_iter<=150) {
    uint32_t nr = 0;
      print_maxima();
      printf("iteration %d,%e,%e,%e,\n",
             _iter,_pLikelihood,_qLikelihood,_pLikelihood-_qLikelihood);
      printf("Rating ELBO, %e,%e\n",_ratingELBO,_ratingLikelihood);
       printf("Rating Posteriors, %e,%e\n",_ratingPost,_ratingPost2);
      fflush(stdout);

      fprintf(tf,"%d\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",_iter,_pLikelihood,_qLikelihood,_pLikelihood-_qLikelihood,_ratingELBO,_ratingLikelihood,_ratingPost,_ratingPost2,_ratingPost3);
      fflush(tf);
    
    for (uint32_t n = 0; n < _n; ++n) {
      
      const double  **elogtheta = _Elogtheta.const_data();
      const double  **elogbeta = _Elogbeta.const_data();
      const vector<uint32_t> *movies = _ratings.get_movies(n);
      
      for (uint32_t j = 0; j < movies->size(); ++j) {
      	uint32_t m = (*movies)[j];
      	yval_t y = _ratings.r(n,m);
      	
      	phi.zero();
      	for (uint32_t k = 0; k < _k; ++k)
      	  phi[k] = elogtheta[n][k] + elogbeta[m][k];
      	phi.lognormalize();
      	if (y > 1)
      	  phi.scale(y);

      	if (n == 0)
      	  lerr("(%d:%d) adding phi %s", n, m, phi.s().c_str());
      	
      	_anext.add_slice(n, phi);
      	_cnext.add_slice(m, phi);
      	nr++;
      	if (nr % 100000 == 0)
      	  lerr("iter:%d ratings:%d total:%d frac:%.3f", 
      	       _iter, nr, _ratings.nratings(), 
      	       (double)nr / _ratings.nratings());
      }
     
    }

    if (true)
    {// update all thetas and betas in parallel
      lerr("updating bnext and dnext");
      _bnext.add_to(_ebeta_sum);
      _dnext.add_to(_etheta_sum);
      lerr("done updating dnext");
      
      _acurr.swap(_anext);
      _bcurr.swap(_bnext);
      _ccurr.swap(_cnext);
      _dcurr.swap(_dnext);
      
      lerr("done swapping");
      
      update_global_state();
    }

      
    lerr("done updating global state");

    printf("\r iteration %d", _iter);
    fflush(stdout);
    save_model();
      
    if (_iter %10 == 0) {
      lerr("Saving state at iteration %d duration %d secs", _iter, duration());
        printf("Saving Model\n");
        fflush(stdout);
      //auc();
      save_model();
        printf("Model Saved\n");
        fflush(stdout);
        
     // _env.save_state_now = false;
    }

      compute_likelihood();
    if (_iter % _env.reportfreq == 0) {
      lerr("computing validation likelihood");
      //compute_likelihood();
        lerr("ELBO: ", _pLikelihood-_qLikelihood);
      lerr("done computing validation likelihood");
      lerr("saving model");
      //save_model();
      lerr("done saving model");
      //auc();
    }
    _iter++;
  }
    fclose(tf);
} 









void
ZIPF::compute_likelihood()
{
    double logp = 0;
    double logq = 0;
    _miscount = 0;
    
    dpair p_and_q(0,0);
    
    
    p_and_q = rating_likelihood();
    logp += p_and_q.first;
    logq += p_and_q.second;
    
    printf("Micount %d\n", _miscount);
    fflush(stdout);
    _ratingLikelihood = logp;
    _ratingELBO = logp - logq;
    
    /**
    p_and_q = xi_pi_likelihood();
    logp+= p_and_q.first;
    logq += p_and_q.second;
    
    
    p_and_q = psi_likelihood();
    logp+= p_and_q.first;
    logq += p_and_q.second;
    
    logp+= line_7_likelihood();
    logq+= sEntropy;
    logq+= elogPBigPhi;
    logq+= elogQBigPhi;
    **/
    _pLikelihood = logp;
    _qLikelihood = logq;
}

void
ZIPF::observedPosteriorLikelihood()
{
    _obsPosterior = 0;
    for (uint32_t n = 0; n < _n; ++n)
        for (uint32_t m=0; m < _m;++m)
        {
            _obsPosterior+=pair_likelihood(n,m,_ratings.r(n,m));
        }
    
}

dpair
ZIPF:: rating_likelihood()
{
    Array phi(_k);
    double logp = 0;
    double logq = 0;
    _ratingPost = 0;
    _ratingPost3 = 0;
    _ratingPost2 = 0;

    const double ** etheta = _Etheta.const_data();
    const double ** elogtheta = _Elogtheta.const_data();
    const double ** ebeta = _Ebeta.const_data();
    const double ** elogbeta = _Elogbeta.const_data();
    
    const double ** const ad = _acurr.const_data();
    const double ** const bd = _bcurr.const_data();
    const double ** const cd = _ccurr.const_data();
    const double ** const dd = _dcurr.const_data();
    
    double zeroLikelihood = 0;
    for (uint32_t n = 0; n < _n; ++n)
    {
        //const vector<uint32_t> *movies = _ratings.get_movies(n);
        //for (uint32_t j = 0; j < movies->size(); ++j)
        //{
        for (uint32_t m = 0; m < _m; m++)
        {
           // uint32_t m = (*movies)[j];
            yval_t y = _ratings.r(n,m);
            if (y>0)
             logp += pair_likelihood(n,m,y);
            if (y == 0)
                zeroLikelihood+=pair_likelihood(n, m, 0);
            
            
           // if (y== 0 && _ratings.contains(n, m))

            if (y!= 0)
            {
            phi.zero();
            for (uint32_t k = 0; k < _k; ++k)
                phi[k] = elogtheta[n][k] + elogbeta[m][k];
            phi.lognormalize();
            
            double v = .0;
            for (uint32_t k = 0; k < _k; ++k)
                v +=  phi[k] * log(phi[k]);
            logq += v;
            }
            
        }
    }
    _ratingPost = logp;
    _ratingPost2 = logp - logq;
    _ratingPost3 = logp+zeroLikelihood;
    logp+= zeroLikelihood;
    
    
    for (uint32_t n = 0; n < _n; ++n)
    {
        // likelihoods for gamma
        for (uint32_t k = 0; k < _k; ++k) {
            logp += _a * log(_b) + (_a - 1) * elogtheta[n][k];
            logp -= _b * etheta[n][k] + gsl_sf_lngamma(_a);
        }
        for (uint32_t k = 0; k < _k; ++k) {
            logq += ad[n][k] * log(bd[n][k]) + (ad[n][k] - 1) * elogtheta[n][k];
            logq -= bd[n][k] * etheta[n][k] - gsl_sf_lngamma(ad[n][k]);
        }
    }
    
    for (uint32_t m = 0; m < _m; ++m)  {
        for (uint32_t k = 0; k < _k; ++k) {
            logp += _c * log(_d) + (_c - 1) * elogbeta[m][k];
            logp -= _d * ebeta[m][k] + gsl_sf_lngamma(_c);
        }
        for (uint32_t k = 0; k < _k; ++k) {
            logq += cd[m][k] * log(dd[m][k]) + (cd[m][k] - 1) * elogbeta[m][k];
            logq -= dd[m][k] * ebeta[m][k] + gsl_sf_lngamma(cd[m][k]);
        }
        
    }
    
    return dpair(logp,logq);
}




uint32_t
ZIPF::factorial(uint32_t y) const
{
    uint32_t output=1;
    for (uint32_t i = 1; i <= y; ++i)
    {
        output*=i;
    }
    return output;
    
}

double
ZIPF::pair_likelihood(uint32_t p, uint32_t q, yval_t y) const
{
    const double **etheta = _Etheta.const_data();
    const double **ebeta = _Ebeta.const_data();
    double s = .0;
    for (uint32_t k = 0; k < _k; ++k)
        s += etheta[p][k] * ebeta[q][k];
    if (s < 1e-30)
        s = 1e-30;
    info("%d, %d, s = %f, f(y) = %ld\n", p, q, s, factorial(y));
    if (y>0)
        return y * log(s) - s;// - log(factorial(y));
    if (batch_infer1)
        return log( 1 + _s.at(p,q) * ( exp(-1*s) -1 ));
    return -s;
//    return 0;
    }

dpair
ZIPF:: xi_pi_likelihood()
{
    double logp = 0;
    double logq = 0;
    
    
    const double ** elogpiI = _eLogPiI.const_data();
    const double ** elogpiU = _eLogPiU.const_data();
    const double ** gammaI = _gammaI.const_data();
    const double ** gammaU = _gammaU.const_data();


    for (uint32_t n = 0; n < _n; ++n)
    {
        
        // likelihoods for pi
        double gammaSum = 0;
        for (uint32_t kk = 0; kk < _k2; ++kk)
        {
            logp += (_alpha[kk]-1)*elogpiU[n][kk]
            + gsl_sf_lngamma(_alpha[kk]);
            double d = _gammaU.at(n,kk);
            
            logq += (gammaU[n][kk]-1)*elogpiU[n][kk]
            + gsl_sf_lngamma(gammaU[n][kk]);
            gammaSum += gammaU[n][kk];
        }
        logp -= gsl_sf_lngamma(_alpha0);
        logq -= gsl_sf_lngamma(gammaSum);
    
        
        // likelihoods for lambda
        logp += log(_sigma1U) + ( SQR(_lambdaU[n])
                                 +SQR(_sigma_xiU))*(-1/(2*SQR(_sigma1U)));
        logp += (1/2)*log(2*M_E*M_PI*SQR(_sigma_xiU));
        
    }
    
    for (uint32_t m = 0; m < _m; ++m)  {
        double gammaSum = 0;
        for (uint32_t kk = 0; kk < _k2; ++kk)
        {
            logp += (_alpha[kk]-1)*elogpiI[m][kk]
            + gsl_sf_lngamma(_alpha[kk]);
           
            logq += (gammaI[m][kk]-1)*elogpiI[m][kk]
            + gsl_sf_lngamma(gammaI[m][kk]);
            gammaSum += gammaI[m][kk];
        }
        logp-= gsl_sf_lngamma(_alpha0);
        logq -= gsl_sf_lngamma(gammaSum);
        
        // likelihoods for lambda
        logp += log(_sigma1I) + ( SQR(_lambdaI[m])
                                 +SQR(_sigma_xiI))*(-1/(2*SQR(_sigma1I)));
        logp += (1/2)*log(2*M_E*M_PI*SQR(_sigma_xiI));
    }
    return dpair(logp, logq);
    
}

dpair
ZIPF:: psi_likelihood()
{
    double logp = 0;
    double logq = 0;
    double mu;
    double sig2 = SQR(_sigma_psi);
    for (uint32_t kk = 0; kk < _k2; ++kk)
    {
        mu = _mu[kk];
        logp += _mu0 * log(_sigma0)/SQR(_sigma0);
        logp += ( SQR(mu)+sig2)*(-1/(2*SQR(_sigma0)));
        logp += SQR(_mu0)/(2*_sigma0)-log(_sigma0);
        
        logq += (1/2)*log(2*M_E*M_PI*SQR(_sigma_psi));
    }
    return dpair(logp,logq);

}

double
ZIPF::line_7_likelihood()
{
const double ** const sd = _s.const_data();
const double *** const phid = _bigPhi.const_data();
const double ** const td = _t.const_data();
 double logp = 0;
 for (uint32_t n = 0; n < _n; ++n)
 {
     for (uint32_t m = 0; m < _m; ++m)
     {
         double xVal = _lambdaI[m]+_lambdaU[n];
         for (uint32_t kk = 0; kk < _k2; ++kk)
             xVal += _mu[kk]*phid[n][m][kk];
         logp += xVal*sd[n][m];
         
         double val = _eExpXiU[n]*_eExpXiI[m]*td[n][m];
         logp-= -log(1+val);
     }
 }
 return logp;
}

double
ZIPF::termination_likelihood()
{
    uint32_t k = 0, kzeros = 0, kones = 0;
    double s = .0, szeros = 0, sones = 0;
    
    CountMap *mp = NULL;
    FILE *ff = NULL;

    mp = &_validation_map;
    
    
    for (CountMap::const_iterator i = mp->begin();
         i != mp->end(); ++i) {
        const Rating &e = i->first;
        uint32_t n = e.first;
        uint32_t m = e.second;
        
        yval_t r = i->second;
        double u =  pair_likelihood(n,m,r);
        s += u;
        k += 1;
    }
    info("s = %.5f\n", s);
    fprintf(ff, "%d\t%d\t%.9f\t%d\n", _iter, duration(), s / k, k);
    fflush(ff);
    double a = s / k;
    
   // if (!validation)
   //     return;
    
    bool stop = false;
    int why = -1;
    if (_iter > 10) {
        if (a > _prev_h && _prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.00001) {
            stop = true;
            why = 0;
        } else if (a < _prev_h)
            _nh++;
        else if (a > _prev_h)
            _nh = 0;
        
        if (_nh > 3) { // be robust to small fluctuations in predictive likelihood
            why = 1;
            stop = true;
        }
    }
    _prev_h = a;
    FILE *f = fopen(Env::file_str("/max.txt").c_str(), "w");
    fprintf(f, "%d\t%d\t%.5f\t%d\n", 
            _iter, duration(), a, why);
    fclose(f);
   // if (stop) {
    //    do_on_stop();
     //   exit(0);
    //}
    return 0;
}

void
ZIPF::print_maxima()
{
    int maxX = 0;
    int maxY = 0;
    double maxGammaU = 0;
    double maxGammaI = 0;
    
    /**
    for (uint32_t n = 0; n < _n; n++)
        for (uint32_t k = 0; k < _k2; k++)
            if(_gammaU.at(n,k)>maxGammaU)
            {
                maxGammaU = _gammaU.at(n,k);
                maxX = n;
                maxY = k;
            }
    printf("Max Gamma U %e,%d,%d\n",
           maxGammaU,maxX,maxY);
    fflush(stdout);
    
    for (uint32_t m = 0; m < _m; m++)
        for (uint32_t k = 0; k < _k2; k++)
            if(_gammaI.at(m,k)>maxGammaI)
            {
                maxGammaI = _gammaI.at(m,k);
                maxX = m;
                maxY = k;
            }
    
    printf("Max Gamma I %e,%d,%d\n",
           maxGammaI,maxX,maxY);
    fflush(stdout);
    
    double maxT = 0;
    double maxL = 0;
    double minS = 1;
    double maxS = 0;
    double maxPhi = 0;
    
    for (uint32_t n = 0; n < _n; ++n)
        for (uint32_t m = 0; m < _m; ++m)
            if (maxT < _t.at(n,m))
            {
                maxT = _t.at(n, m);
                maxX = n;
                maxY = m;
            }
    printf("Max T %e,%d,%d\n",
           maxT,maxX,maxY);
    fflush(stdout);
    
    for (uint32_t n = 0; n < _n; ++n)
        for (uint32_t m = 0; m < _m; ++m)
            if (maxL < _l.at(n,m))
            {
                maxL = _l.at(n, m);
                maxX = n;
                maxY = m;
            }
    printf("Max L %e,%d,%d\n",
           maxL,maxX,maxY);
    fflush(stdout);
    
    for (uint32_t n = 0; n < _n; ++n)
        for (uint32_t m = 0; m < _m; ++m)
            
            if (maxS < _s.at(n,m))
            {
                maxS = _s.at(n, m);
                maxX = n;
                maxY = m;
            }
    printf("Max S %e,%d,%d\n",
           maxS,maxX,maxY);
    fflush(stdout);
    
    
    for (uint32_t n = 0; n < _n; ++n)
        for (uint32_t m = 0; m < _m; ++m)
            if (minS > _s.at(n,m))
            {
                minS = _s.at(n, m);
                maxX = n;
                maxY = m;
            }
    
    printf("Min S %e,%d,%d\n",
           minS,maxX,maxY);
    fflush(stdout);
    
    for (uint32_t n = 0; n < _n; ++n)
        for (uint32_t m = 0; m < _m; ++m)
            for (uint32_t k = 0; k < _k2; ++k)
            if (maxPhi > _bigPhi.at(n,m,k))
            {
                maxPhi = _bigPhi.at(n,m,k);
                maxX = n;
                maxY = m;
            }
    printf("Max Phi %e,%d,%d\n",
           maxPhi,maxX,maxY);
    fflush(stdout);
    
    
    
    double maxLambdaU = 0;
    double maxLambdaI = 0;
    for (uint32_t n = 0; n < _n; ++n)
            if (maxLambdaU < _lambdaU[n])
            {
                maxLambdaU = _lambdaU[n];
                maxX = n;
            }
    printf("Max LambdaU %e,%d\n",
           maxLambdaU,maxX);
    fflush(stdout);
    for (uint32_t m = 0; m < _m; ++m)
        if (maxLambdaI < _lambdaI[m])
        {
            maxLambdaI = _lambdaI[m];
            maxX = m;
        }
    

    printf("Max LambdaI %e,%d\n",
       maxLambdaI,maxX);
    fflush(stdout);
    
    
    double minLambdaU = 1;
    double minLambdaI = 1;
    for (uint32_t n = 0; n < _n; ++n)
        if (minLambdaU >  _lambdaU[n])
        {
            minLambdaU = _lambdaU[n];
            maxX = n;
        }
    printf("Min LambdaU %e,%d\n",
           minLambdaU,maxX);
    fflush(stdout);
    for (uint32_t m = 0; m < _m; ++m)
        if (minLambdaI > _lambdaI[m])
        {
            minLambdaI = _lambdaI[m];
            maxX = m;
        }
    
    
    printf("Min LambdaI %e,%d\n",
           minLambdaI,maxX);
    fflush(stdout);
    
    double maxExpU = 0;
    double maxExpI = 0;
    for (uint32_t n = 0; n < _n; ++n)
        if (maxExpU < _eExpXiU[n])
        {
            maxExpU = _eExpXiU[n];
            maxX = n;
        }
    printf("Max Exp Xi U %e,%d\n",
           maxExpU,maxX);
    fflush(stdout);
    for (uint32_t m = 0; m < _m; ++m)
        if (maxExpI < _eExpXiI[m])
        {
            maxExpI = _eExpXiI[m];
            maxX = m;
        }
    
    
    printf("Max Exp Xi I %e,%d\n",
           maxExpI,maxX);
    fflush(stdout);
    
    double maxMu = 0;
    for (uint32_t k = 0; k < _k2; ++k)
        if (maxMu < _mu[k])
        {
            maxMu = _mu[k];
            maxX = k;
        }
    printf("Max Mu %e,%d\n",
           maxMu,maxX);
    fflush(stdout);
    **/
    
    double maxEtheta = 0;
    for (uint32_t n = 0; n < _n; ++n)
        for (int k = 0; k < _k; ++k)
            if (maxEtheta < _Etheta.at(n,k))
            {
                maxEtheta = _Etheta.at(n,k);
                maxX = n;
                maxY = k;
            }
    printf("Max Etheta %e,%d,%d\n",
           maxEtheta,maxX,maxY);
    fflush(stdout);
    
    
    double maxEbeta = 0;
    for (uint32_t n = 0; n < _m; ++n)
        for (int k = 0; k < _k; ++k)
            if (maxEbeta < _Ebeta.at(n,k))
            {
                maxEbeta = _Ebeta.at(n,k);
                maxX = n;
                maxY = k;
            }
    printf("Max Ebeta %e,%d,%d\n",
           maxEbeta,maxX,maxY);
    fflush(stdout);
   // FILE *maxf = fopen(Env::file_str("/max.stv").c_str(), "w");

    //fprintf(maxf,"%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",maxGammaU,maxGammaI,maxT,maxL,maxS,minS,maxPhi,maxLambdaU,maxLambdaI,maxMu,maxExpU,maxExpI,maxEtheta,maxEbeta);
    //fflush(maxf);
    //fclose(maxf);
    
    

}


void
ZIPF::save_model()
{
    save_item_state("/Users/msimchowitz/Documents/JP/ebeta1.txt", _Ebeta);
    //save_item_state("/c.txt", _ccurr);
    //save_item_state("/d.txt", _dcurr);
    
    save_user_state("/Users/msimchowitz/Documents/JP/etheta1.txt", _Etheta);
   // save_user_state("/a.txt", _acurr);
   // save_user_state("/b.txt", _bcurr);

    //save_state("/u.txt", _ucurr);
    //save_state("/Eu.txt", _Eu);
    //save_state("/Ei.txt", _Ei);
    
}



void
ZIPF::save_user_state(string s, const Matrix &mat)
{
    FILE * tf = fopen(s.c_str(), "w");
    assert(tf);
    const double **gd = mat.data();
    for (uint32_t i = 0; i < _n; ++i) {
        const IDMap &m = _ratings.seq2user();
        IDMap::const_iterator idt = m.find(i);
        if (idt != m.end()) {
            fprintf(tf,"%d\t", i);
            debug("looking up i %d\n", i);
            fprintf(tf,"%d\t", (*idt).second);
            for (uint32_t k = 0; k < _k; ++k) {
                if (k == _k - 1)
                    fprintf(tf,"%.5f\n", gd[i][k]);
                else
                    fprintf(tf,"%.5f\t", gd[i][k]);
            }
        }
    }
    fclose(tf);
}

void
ZIPF::save_item_state(string s, const Matrix &mat)
{
    FILE *tf = fopen(s.c_str(), "w");
    assert(tf);
    const double **cd = mat.data();
    for (uint32_t i = 0; i < _m; ++i) {
        const IDMap &m = _ratings.seq2movie();
        IDMap::const_iterator idt = m.find(i);
        if (idt != m.end()) {
            fprintf(tf,"%d\t", i);
            debug("looking up i %d\n", i);
            fprintf(tf,"%d\t", (*idt).second);
            for (uint32_t k = 0; k < _k; ++k) {
                if (k == _k - 1)
                    fprintf(tf,"%.5f\n", cd[i][k]);
                else
                    fprintf(tf,"%.5f\t", cd[i][k]);
            }
        }
    }
    fclose(tf);
}


void
ZIPF::save_state(string s, const Array &mat)
{
    FILE *tf = fopen(Env::file_str(s.c_str()).c_str(), "w");
    const double *cd = mat.data();
    for (uint32_t k = 0; k < mat.size(); ++k) {
        if (k == _k - 1)
            fprintf(tf,"%.5f\n", cd[k]);
        else
            fprintf(tf,"%.5f\t", cd[k]);
    }
    fclose(tf);
}


void
ZIPF::batch_infer2()
{
    batch_infer1 = false;
    printf("initializing \n");
    fflush(stdout);
    initialize();
    printf("done initializing \n");
    fflush(stdout);
    info("ZIPF initialization done\n");
    
    FILE *tf = fopen("/Users/msimchowitz/Documents/JP/elbo1.tsv", "w");
    
    assert(tf);
    //approx_log_likelihood();
    fflush(stdout);
    
    Array phi(_k);
    while (_iter<150) {
        uint32_t nr = 0;
        print_maxima();
        printf("iteration %d,%e,%e,%e,\n",
               _iter,_pLikelihood,_qLikelihood,_pLikelihood-_qLikelihood);
        fflush(stdout);
        printf("more ELBO, %e,%e\n",_ratingELBO,_ratingLikelihood);
        fprintf(tf,"%d\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",_iter,_pLikelihood,_qLikelihood,_pLikelihood-_qLikelihood,_ratingELBO,_ratingLikelihood,_ratingPost,_ratingPost2,_ratingPost3);
        fflush(tf);

        
        for (uint32_t n = 0; n < _n; ++n) {
            
            const double  **elogtheta = _Elogtheta.const_data();
            const double  **elogbeta = _Elogbeta.const_data();
            const vector<uint32_t> *movies = _ratings.get_movies(n);
            
            for (uint32_t j = 0; j < movies->size(); ++j) {
                uint32_t m = (*movies)[j];
                yval_t y = _ratings.r(n,m);
                
                phi.zero();
                for (uint32_t k = 0; k < _k; ++k)
                    phi[k] = elogtheta[n][k] + elogbeta[m][k];
                phi.lognormalize();
                if (y > 1)
                    phi.scale(y);
                
                if (n == 0)
                    lerr("(%d:%d) adding phi %s", n, m, phi.s().c_str());
                
                _anext.add_slice(n, phi);
                _cnext.add_slice(m, phi);
                nr++;
                if (nr % 100000 == 0)
                    lerr("iter:%d ratings:%d total:%d frac:%.3f",
                         _iter, nr, _ratings.nratings(),
                         (double)nr / _ratings.nratings());
            }
        }
        
        if (true){
            // update all thetas and betas in parallel
            lerr("updating bnext and dnext");
            _bnext2.add_to(_ebeta_sum2);
            _dnext2.add_to(_etheta_sum2);
            lerr("done updating dnext");
            
            _acurr.swap(_anext);
            _bcurr2.swap(_bnext2);
            _ccurr.swap(_cnext);
            _dcurr2.swap(_dnext2);
            
            lerr("done swapping");
            
            update_global_state2();
        }
        
        lerr("done updating global state");
        
        printf("\r iteration %d", _iter);
        fflush(stdout);
        compute_likelihood();
        if (_iter % _env.reportfreq == 0) {
            lerr("computing validation likelihood");
            //compute_likelihood();
            lerr("ELBO: ", _pLikelihood-_qLikelihood);
            lerr("done computing validation likelihood");
            lerr("saving model");
            //save_model();
            lerr("done saving model");
            //auc();
        }
        _iter++;
    }
    fclose(tf);
    save_model();
    
}

void
ZIPF::update_global_state2()
{
    set_gamma_exp2(_acurr, _bcurr2, _Etheta, _Elogtheta);
    set_gamma_exp2(_ccurr, _dcurr2, _Ebeta, _Elogbeta);
    set_etheta_sum2();
    set_ebeta_sum2();
    
    set_to_prior_users2(_anext, _bnext2);
    set_to_prior_movies2(_cnext, _dnext2);
}

void
ZIPF::set_etheta_sum2()
{
    _etheta_sum2.zero();
    const double **etheta  = _Etheta.const_data();
    for (uint32_t k = 0; k < _k; ++k) {
        for (uint32_t n = 0; n < _n; ++n)
            _etheta_sum2[k] += etheta[n][k];
    }
    
}

void
ZIPF::set_ebeta_sum2()
{
    _ebeta_sum2.zero();
    const double **ebeta  = _Ebeta.const_data();
    for (uint32_t k = 0; k < _k; ++k) {
        for (uint32_t m = 0; m < _m; ++m)
            _ebeta_sum2[k] += ebeta[m][k];
    }
    
}

void
ZIPF::set_gamma_exp2(const Matrix &a, const Array &b, Matrix &v1, Matrix &v2)
{
    const double ** const ad = a.const_data();
    const double * const bd = b.const_data();
    //const double ** const sd = _s.const_data();
    double **vd1 = v1.data();
    double **vd2 = v2.data();
    for (uint32_t i = 0; i < a.m(); ++i)
        for (uint32_t j = 0; j < b.n(); ++j) {
            assert(bd[j]);
            vd1[i][j] = ad[i][j] / bd[j];
            vd2[i][j] = gsl_sf_psi(ad[i][j]) - log(bd[j]);
        }
}




