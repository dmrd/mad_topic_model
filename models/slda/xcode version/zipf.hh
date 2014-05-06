#ifndef ZIPF_HH
#define ZIPF_HH

#include "env.hh"
#include "ratings.hh"
#include <math.h>

class ZIPF {
public:
  ZIPF(Env &env, Ratings &ratings);
  ~ZIPF();

  void batch_infer();
    void batch_infer2();

  


private:
  void initialize();
  void set_to_prior_users(Matrix &a, Matrix &b);
  void set_to_prior_movies(Matrix &a, Matrix &b);
    void load_validation_and_test_sets();
    void print_maxima();
  
  //void initialize_bias();
  //void set_to_prior_biases();
  //void update_global_state_bias();
  //void set_gamma_exp_bias(const Array &u, double v, Array &w1, Array &w2);
  //double link_prob_bias(uint32_t user, uint32_t movie) const;
  //double pair_likelihood_bias(uint32_t p, uint32_t q, yval_t y) const;


  void save_model();
  //void write_movie_list(string label, uint32_t u, const vector<uint32_t> &movies);
  //void write_movie_list(string label, const vector<uint32_t> &movies);

  void save_user_state(string s, const Matrix &mat);
  void save_item_state(string s, const Matrix &mat);
  void save_state(string s, const Array &mat);
    double termination_likelihood();
  //void do_on_stop();

  //void load_file(string s, Matrix &mat);
  //void load_beta_and_theta();
  //void load_file(string s, Array &mat);

    double approx_log_likelihood();
 
  void set_etheta_sum();
  void set_ebeta_sum();;
  //void load_validation_and_test_sets();

  void swapCommunityVar();
  void set_BigPhi();
  void setExpXiandPsi();
  void setLambda();
  void setGamma();
  void set_mu();
  void update_rho();

  void communityInitialize();
  void init_s();
  void init_bigPhi();
  void ratingInitalize();
  int load_gamma();
    void getDegrees();
    
   
    void set_s();
    void set_l();
    void set_t();
    
    void update_global_state2();
    void set_ebeta_sum2();
    void set_etheta_sum2();
    
  void set_gamma_exp(const Matrix &a, const Matrix &b, Matrix &v1, Matrix &v2);
  void set_gamma_exp_init(const Matrix &a, Matrix &v1, Matrix &v2, double v);
    
  //void set_gamma_exp1(const Matrix &a, const Array &b, Matrix &v);
  //void set_gamma_exp2(const Matrix &a, const Array &b, Matrix &v);
  //void compute_etheta_sum(const UserMap &sampled_users);
  //void compute_ebeta_sum(const MovieMap &sampled_movies);
  //void adjust_etheta_sum(const UserMap &sampled_users);
  //void adjust_ebeta_sum(const MovieMap &sampled_movies);
  //void optimize_user_rate_parameters(uint32_t n);
  //void optimize_user_shape_parameters_helper(uint32_t n, 
	//				     Matrix &cphi);
  //void optimize_user_shape_parameters(uint32_t n);
  //void optimize_user_shape_parameters2(uint32_t n);

    // likelihood helper functions
    dpair psi_likelihood();
    dpair xi_pi_likelihood();
    dpair rating_likelihood();
    double line_7_likelihood();
    bool batch_infer1;
    
  //void auc(bool bias = false);
  //double link_prob(uint32_t user, uint32_t movie) const;
    dpair rating_likelihood2();
  void update_global_state();
  double pair_likelihood(uint32_t p, uint32_t q, yval_t y) const;
  double multi_likelihood(uint32_t p, uint32_t q, yval_t y) const;
    
    void set_to_prior_users2(Matrix &a, Array &b);
    void set_to_prior_movies2(Matrix &a, Array &b);

    void set_gamma_exp2(const Matrix &a, const Array &b, Matrix &v1, Matrix &v2);
  //void test_likelihood();
  void compute_likelihood();
    void observedPosteriorLikelihood();
    uint32_t factorial(uint32_t y) const;
  //uint32_t factorial(uint32_t n) const;  

  //void init_heldout();
  //void set_test_sample(int s);
  //void set_training_sample();
  //void set_validation_sample(int s);
  //void get_random_rating1(Rating &r) const;
  //void get_random_rating2(Rating &r) const;
  //void get_random_rating3(Rating &r) const;
  uint32_t duration() const;
  bool rating_ok(const Rating &e) const;
  //bool is_test_rating(const Rating &e) const;
  //void write_sample(FILE *, SampleMap &m);
  //void write_sample(FILE *, CountMap &m);
  //void write_ebeta(uint32_t);
  //void write_etheta(uint32_t);

  Env &_env;
  Ratings &_ratings;

  uint32_t _n;
  uint32_t _m;
  uint32_t _k;
  uint32_t _k2;
  uint32_t _iter;
    uint32_t _miscount;

  // Gamma Priors on user preferences 
  // and item features;
  double _a;
  double _b;
  double _c;
  double _d;
  double _epsilon;
    
    double _s_epsilon;
    double _alpha0;
    double _ratingELBO;
    double _ratingLikelihood;
    double _ratingPost;
    double _ratingPost2;
    double _sigma_xiU2;
    double _sigma_xiI2;
    
    double _ratingPost3;


    bool epsInflated;

    
    double elogPBigPhi;
    double elogQBigPhi;
    double sEntropy;
    double _pLikelihood;
    double _qLikelihood;
  time_t _start_time;
  // current and updated Gamma parameters
  // for latent user preferences and item features
  Matrix _acurr;
  Matrix _bcurr;
  Matrix _ccurr;
  Matrix _dcurr;
  Matrix _anext;
  Matrix _bnext;
  Matrix _cnext;
  Matrix _dnext;
   Array _bcurr2;
   Array _dcurr2;
   Array _bnext2;
   Array _dnext2;
    
    Matrix _acurrP;
    Matrix _bcurrP;
    Matrix _ccurrP;
    Matrix _dcurrP;
    Matrix _anextP;
    Matrix _bnextP;
    Matrix _cnextP;
    Matrix _dnextP;
    Array _bcurr2P;
    Array _dcurr2P;
    Array _bnext2P;
    Array _dnext2P;
    
    Matrix _eNext;
    Matrix _gNext;
    Array _fNext;
    Array _hNext;
    
    Array v1;
    Array v2;
 
    //Latent Community Parameters
    Matrix _Elogxi;
    Matrix _Exi;
    Matrix _Exi_sum;
    
    Matrix _Elogzeta;
    Matrix _Ezeta;
    Matrix _Ezeta_sum;
    
    //Latent Popularity Weights
    Matrix _ElogAlpha1
    Matrix _ElogAlpha2
    Matrix _EAlpha1;
    Matrix _EAlpha2;
    Matrix _EAlpha1_sum;
    Matrix _EAlpha2_sum;
    
  //Matrix _phi;
    
  Matrix _Elogtheta;
  Matrix _Etheta;
  Matrix _Elogbeta;
  Matrix _Ebeta;

  Matrix _etheta_sum;
  Matrix _ebeta_sum;
  Matrix _etheta_sumOld;
  Matrix _ebeta_sumOld;
    
    Array _ebeta_sum2;
    Array _etheta_sum2;
    

  Array _alpha;
  Array _psi;
  Array _xiU;
  Array _xiI;
  Matrix _piU;
  Matrix _piI;
    
    IntArray _userDegree;
    IntArray _itemDegree;


  gsl_rng *_r;
  FILE *_hf;
  FILE *_vf;
  FILE *_tf;
  FILE *_af;
  FILE *_pf;


  CountMap _test_map;
  RatingList _test_ratings;
  CountMap _validation_map;
  RatingList _validation_ratings;
  UserMap _sampled_users;
  UserMap _sampled_movies;

  // learning rates
  double _tau0;
  double _muTau0;
  double _kappa;
  double _muKappa;
  double _rho;
  double _muRho;
  
    uint32_t _maxUserDeg;
    uint32_t _maxItemDeg;
  uint32_t _nh;
  double _prev_h;
  bool _save_ranking_file;
  uArray _itemc;
  bool _use_rate_as_score;
  uint32_t _topN_by_user;
};

inline uint32_t
ZIPF::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline bool
ZIPF::rating_ok(const Rating &r) const
{
  assert (r.first  < _n && r.second < _m);
  const CountMap::const_iterator u = _test_map.find(r);
  if (u != _test_map.end())
    return false;
  const CountMap::const_iterator w = _validation_map.find(r);
  if (w != _validation_map.end())
    return false;
  return true;
}

/**
inline bool
ZIPF::is_test_rating(const Rating &r) const
{
  assert (r.first  < _n && r.second < _m);
  const CountMap::const_iterator u = _test_map.find(r);
  if (u != _test_map.end())
    return true;
  return false;
}
**/


#endif
