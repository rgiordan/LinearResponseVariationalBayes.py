functions {
    real mu_log_prior(real mu, real mu_prior_epsilon,
                      real mu_prior_mean, real mu_prior_var,
                      real mu_prior_mean_c, real mu_prior_var_c) {
                      
        real mu_log_pdf;
        real mu_normal_lpdf_cache;
        real mu_normal_c_lpdf_cache;
        //real mu_student_t_lpdf_cache;

        // See https://groups.google.com/forum/#!category-topic/stan-users/general/_gOPDicnDl0
        mu_normal_lpdf_cache = normal_lpdf(mu | mu_prior_mean, mu_prior_var);
        mu_normal_c_lpdf_cache = normal_lpdf(mu | mu_prior_mean_c, mu_prior_var_c);
        //mu_student_t_lpdf_cache = student_t_lpdf(mu | mu_prior_t, mu_prior_mean, sqrt(mu_prior_var));

        // Express the mu prior as a mixture of a normal and t prior.
        if (mu_prior_epsilon == 0) {
          mu_log_pdf = normal_lpdf(mu | mu_prior_mean, mu_prior_var);
        } else if (mu_prior_epsilon == 1) {
          mu_log_pdf = normal_lpdf(mu | mu_prior_mean_c, mu_prior_var_c);
          // mu ~ student_t(mu_prior_t, mu_prior_mean, sqrt(mu_prior_var));
        } else {
          // It is a mixture.
          mu_log_pdf = log_sum_exp(log(1 - mu_prior_epsilon) + mu_normal_lpdf_cache,
                                   log(mu_prior_epsilon) + mu_normal_c_lpdf_cache);
        }
        return mu_log_pdf;
    }
    
    real log_prior(real tau, vector beta, real mu,
                   vector beta_prior_mean, matrix beta_prior_var,
                   real tau_prior_alpha, real tau_prior_beta,
                   real mu_prior_epsilon,
                   real mu_prior_mean, real mu_prior_var,
                   real mu_prior_mean_c, real mu_prior_var_c) {
        real log_prior;
        log_prior = 0;
        log_prior = log_prior + gamma_lpdf(tau | tau_prior_alpha, tau_prior_beta);
        log_prior = log_prior + multi_normal_lpdf(beta | beta_prior_mean, beta_prior_var);
        log_prior = log_prior + mu_log_prior(mu, mu_prior_epsilon,
                                             mu_prior_mean, mu_prior_var,
                                             mu_prior_mean_c, mu_prior_var_c);
        return log_prior;
    }
}

data {
  // Data
  int <lower=0> NG;  // number of groups
  int <lower=0> N;  // total number of observations
  int <lower=0> K;  // dimensionality of parameter vector which is jointly distributed
  int <lower=0, upper=1> y[N];       // outcome variable of interest
  vector[K] x[N];       // Covariates
  
  // y_group is zero-indexed group indicators
  int y_group[N];
  
  // Prior parameters
  matrix[K,K] beta_prior_var;
  vector[K] beta_prior_mean;
  real mu_prior_mean;
  real <lower=0> mu_prior_var;
  real <lower=0> tau_prior_alpha;
  real <lower=0> tau_prior_beta;
  
  // An alternative prior for the mu prior distribution.
  real <lower=0, upper=1> mu_prior_epsilon;
  real mu_prior_mean_c;
  real <lower=0> mu_prior_var_c;
  real <lower=0> mu_prior_t;
}

parameters {
  // Global regressors.
  vector[K] beta;
  
  // The mean of the random effect.
  real mu;

  // The information of the random effect.
  real <lower=0> tau;

  // The actual random effects.
  vector[NG] u;

}

transformed parameters {
  // Latent probabilities
  // vector[N] p;
  // vector[N] logit_p;
  // for (n in 1:N) {
  //   // y_group is zero-indexed, but stan is one-indexed
  //   logit_p[n] = x[n]' * beta + u[y_group[n] + 1];
  //   p[n] = inv_logit(logit_p[n]);
  // }
}

model {
  // priors
  tau ~ gamma(tau_prior_alpha, tau_prior_beta);
  beta ~ multi_normal(beta_prior_mean, beta_prior_var);
  target += log_prior(tau, beta, mu, beta_prior_mean, beta_prior_var,
                      tau_prior_alpha, tau_prior_beta,
                      mu_prior_epsilon, mu_prior_mean, mu_prior_var,
                      mu_prior_mean_c, mu_prior_var_c);
  
  // The model
  for (g in 1:NG) {
    u[g] ~ normal(mu, 1 / tau);
  }

  for (n in 1:N) {
    // y[n] ~ bernoulli(p[n]);
    // y_group is zero-indexed, but stan is one-indexed
    y[n] ~ bernoulli(inv_logit(x[n]' * beta + u[y_group[n] + 1]));
  }
}
