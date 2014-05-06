//
//  mixbase.hh
//  RZIP
//
//  Created by Max Simchowitz on 3/17/14.
//
//

#ifndef RZIP_mixbase_hh
#define RZIP_mixbase_hh

class GaussianMixture2D  {
public:
    GPMatrix(string name, Array m0, Matrix L0,
             uint32_t n, uint32_t k, double b0,
             gsl_rng **r):
    GPBase<Matrix>(name),

    _n(n), _k(k),
    _beta0(b0), _nu0(n0), 
    _x(n,k),
    _elogLambda(n),
    _elogPi(n,k),_rk(n,k),
    _l0Inverse(3),
    _l0(3),
    _m0(2),

    _lInverse(k3,3),_L(k,3), _m(k,2), _nu(k),_beta(k),

    _Nscale(k), _xBar(k,2), _sBar(k,3);

    { }
    
    uint32_t n() const { return _n;}
    uint32_t k() const { return _k;}
    
    void save() const;
    void load();
    
    const Matrix &rk() const         { return _rk; }
    
    // parameter optimization       
    void m_step1();
    void m_step2();
    void uninvert();
    void mstep();

    // estep
    void elogLambda();
    void set_rk();

    // matrix utilities
    double qform(const Array &a, double x1, double x2);
    double invert(const Array &a, Array &b);
    double det(const Array &a);

    void initialize();
    void initialize_exp();
    void save_state(const IDMap &m) const;
    void load_from_lda(string dir, double alpha, uint32_t K);
        
private:
    uint32_t _n;
    uint32_t _k;
    gsl_rng **_r;

    Array _elogLambda;
    Matrix _elogPi;
    // data
    Matrix _x;
    Matrix _rk;

    // distribution priors;
    double _nu0;
    double _beta0;
    Array _l0Inverse;
    Array _l0;
    Array _m0;

    Matrix _lInverse;
    Matrix _L;
    Matrix _m;
    Matrix _nu;
    Matrix _beta;

    // m step terms
    Array _Nscale;
    Array _xBar;
    Matrix _sBar;
};

inline void 
GaussianMixture2D::elogLambda()
{
    Array lSlice(3);
    double * eld = _elogLambda.data();
    const double * nuk = _nu.const_data;

    for (uint32_t k = 0; k < _k; ++k)
    {
        _L.slice(0,k,lSlice);
        eld[k] = gsl_sf_psi(nuk/2)+gsl_sf_psi((nuk-1)/2)
            +log(det(lSlice));
    }
}

inline void
GaussianMixture2D::set_rk()
{
    const double * eld = elogLambda.const_data();
    const double ** elp = elogPi.const_data();
    const double * beta = _beta.const_data();
    const double * nud  = _nu.const_data();

    const double ** xd = _x.const_data();
    const double ** md = _m.const_data();

    double x1,x2; 
    for (uint32_t i = 0; i < _n; ++i)
    {
        Array rkTemp(_k);
        double * rkd = rkTemp.data();
        for (uint32_t k = 0; k < _k; ++k)
        {
            _L.slice(0,k,lSlice);
            rkd[k] = elp[i][k]+eld[k] -(1/beta[k]);
            x1 = xd[i][0] - md[k][0];
            x2 = xd[i][1] - md[k][1];
            rkd[k]-= (nud[k]/2)*qform(lSlice,x1,x2);
        }
        rkTemp.lognormalize();
        _rk.set_elements(i, rkTemp);
    }
}

inline double
GaussianMixture2D::qform(const Array &a, double x1, double x2)
{
    assert(a.size() == 3);
    const double ad = a.const_data();
    return (ad[0]*x1*x1)+(2*ad[1]*x1*x2)+(ad[2]*x2*x2);
}

inline void 
GaussianMixture2D::m_step1()
{
    
    _Nscale.zero();
    _xBar.zero();
    _sBar.zero();

    double * nk = _Nscale.data();
    double ** xb = _xBar.data();
    double ** sl = _sBar.data();
    double ** x = _x.data();
    double scalar, x1,x2;

    for (uint32_t i = 0; i< _n, ++i)
        for (uint32_t k = 0; k< _k;++k)
            nk[k] += r[i][k];
    for (uint32_t i = 0; i< _n, ++i)
        for (uint32_t k = 0; k< _k;++k)
        {
            xb[k][0] += r[i][k]*x[i][0]/nk[k];
            xb[k][1] += r[i][k]*x[i][1]/nk[k];
        }
     for (uint32_t i = 0; i< _n, ++i)
        for (uint32_t k = 0; k< _k;++k)
        {
            scalar = r[i][k]/nk[k];
            x1 = x[i][0]-xb[i][0];
            x2 = x[i][1]-xb[i][0];
            sl[k][0] += x1*x1;
            sl[k][1] += x1*x2;
            sl[k][2] += x2*x2;
        }
}

inline void
GaussianMixture2D::m_step2()
{    
    double * beta = _beta.data();
    double ** m = _m.data();
    double ** linv = _lInverse.data();
    double * nuk = _nu.data();

    const double * nk = _Nscale.const_data();
    const double ** xb = _xBar.const_data():
    const double ** sk = _sBar.const_data():
    const double * l0Inv = _l0Inverse.const_data():

    double x1,x2,scale;

    for (uint32_t k =0; k <_k; ++k)
    {
        
        beta[k] = _b0 + nk[k];
        m[k][0] = (_b0*_m0[0]+nk[k]*xb[k][0])/beta[k];
        m[k][1] = (_b0*_m0[1]+nk[k]*xb[k][1])/beta[k];

        x1 = xb[k][0]-_m0[0];
        x2 = xb[k][1] - _m0[1];
        scale = (_b0*nk[k])/(_b0+nk[k]);

        linv[k][0] = l0Inv[0] + nk[k]*sk[k][0]
        + scale*x1*x1;
        linv[k][1] = l0Inv[1] + nk[k]*sk[k][1]
        + scale*x1*x2;
        linv[k][2] = l0Inv[1] + nk[k]*sk[k][1]
        + scale*x1*x2;

        nu[k] = _nu0 + nk[k]+1;
    }
}

inline void
GaussianMixture2D::uninvert()
{
    Array lSlice(3);
    Array inv(3);

    for (uint32_t k = 0; k < _k; ++k)
    {
        _L.slice(0,k,lSlice);
        invert(lSlice,inv);
        _lInverse.set_elements(k,inv);
    }
}

inline double
GaussianMixture2D::det(const Array &a)
{
    const double * ad = a.const_data();
    assert (a.size() == 3);
    return ad[0]*ad[2]-(ad[1]*ad[1]);
}

inline void
GaussianMixture2D::invert(const Array &a, Array &b)
{
    const double * ad = a.const_data();
    assert (a.size() == 3);
    assert (b.size() == 3);
    double det = det(a);
    assert (det > 0);

    double * bd = b.data();
    bd[0]=a[2]/det;
    bd[1] = -1*a[1]/det;
    bd[2] = a[0]/det;
}

inline void
GaussianMixture2D::mstep()
{
    m_step1();
    m_step2();
    uninvert();
}

inline void
GaussianMixture2D::estep()
{
    elogLambda();
    set_rk();
}



inline void
GPMatrix::initialize()
{
    double **ad = _scurr.data();
    double **bd = _rcurr.data();
    for (uint32_t i = 0; i < _n; ++i)
        for (uint32_t k = 0; k < _k; ++k)
            ad[i][k] = _sprior + 0.01 * gsl_rng_uniform(*_r);
    
    for (uint32_t k = 0; k < _k; ++k)
        bd[0][k] = _rprior + 0.1 * gsl_rng_uniform(*_r);
    
    for (uint32_t i = 0; i < _n; ++i)
        for (uint32_t k = 0; k < _k; ++k)
            bd[i][k] = bd[0][k];
    set_to_prior();
}


/**
inline void
GPMatrix::save_state(const IDMap &m) const
{
    string expv_fname = string("/") + name() + ".tsv";
    string shape_fname = string("/") + name() + "_shape.tsv";
    string rate_fname = string("/") + name() + "_scale.tsv";
    _scurr.save(Env::file_str(shape_fname), m);
    _rcurr.save(Env::file_str(rate_fname), m);
    _Ev.save(Env::file_str(expv_fname), m);
}
**/

/**
inline void
GPMatrix::load()
{
    string shape_fname = name() + "_shape.tsv";
    string rate_fname = name() + "_scale.tsv";
    _scurr.load(shape_fname);
    _rcurr.load(rate_fname);
    compute_expectations();
}
**/


/**
GPMatrix::load_from_lda(string dir, double alpha, uint32_t K)
{
    char buf[1024];
    sprintf(buf, "%s/lda-fits/%s-lda-k%d.tsv", dir.c_str(), name().c_str(), K);
    lerr("loading from %s", buf);
    _Ev.load(buf, 0);
    double **vd1 = _Ev.data();
    double **vd2 = _Elogv.data();
    
    Array b(_k);
    for (uint32_t i = 0; i < _n; ++i) {
        double s = _Ev.sum(i) + alpha * _k;
        for (uint32_t k = 0; k < _k; ++k) {
            double counts = vd1[i][k];
            vd1[i][k] = (alpha + counts) / s;
            vd2[i][k] = gsl_sf_psi(alpha + counts) - gsl_sf_psi(s);
        }
    }
    IDMap m;
    string expv_fname = string("/") + name() + ".tsv";
    _Ev.save(Env::file_str(expv_fname), m);
}
**/
}





#endif
