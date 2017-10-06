library(dplyr)
library(ggplot2)
library(reshape2)
library(jsonlite)
library(mvtnorm)
library(boot) # for inv.logit
library(rstansensitivity)

library(reticulate)
use_python("/usr/bin/python3")

library(lme4)
library(rstan)

rstan_options(auto_write=FALSE)

project_directory <- file.path(
  Sys.getenv("GIT_REPO_LOC"),
  "LinearResponseVariationalBayes.py/Models/LogisticGLMM")
data_directory <- file.path(project_directory, "data/")

# 2, 5, 10, 20, 40, 60
num_obs_per_group <- 10
analysis_name <- sprintf("simulated_data_for_refit_%d", num_obs_per_group)


#######################
# Load data from python

InitializePython()
py_main <- reticulate::import_main()

# Results evaluating the jacknnife
python_jackknife_filename <- file.path(
  data_directory,
  paste(analysis_name, "_python_refit_jackknife_results.pkl", sep=""))

reticulate::py_run_string(
  "
pkl_file = open('" %_% python_jackknife_filename %_% "', 'rb')
vb_jackknife_results = pickle.load(pkl_file)
pkl_file.close()
")

names(py_main$vb_jackknife_results)

reticulate::py_run_string("
import VariationalBayes.SparseObjectives as obj_lib
from scikits.sparse.cholmod import cholesky
import numpy as np
")

reticulate::py_run_string("
true_params = vb_jackknife_results['true_params']
x_mat = vb_jackknife_results['x_mat']
y_g_vec = vb_jackknife_results['y_g_vec']
y_vec = vb_jackknife_results['y_vec']
prior_par = logit_glmm.get_default_prior_params(true_params.beta_dim)
")



########################
# Run stan

k_reg <- ncol(py_main$x_mat)

stan_dat <- list(NG = max(py_main$y_g_vec) + 1,
                 N = length(py_main$y_vec),
                 K = k_reg,
                 y_group = py_main$y_g_vec,
                 y = py_main$y_vec,
                 x = py_main$x_mat,
                 
                 # Priors
                 beta_prior_mean = py_main$prior_par$param_dict$beta_prior_mean$get(),
                 beta_prior_info = py_main$prior_par$param_dict$beta_prior_info$get(),
                 mu_prior_mean = py_main$prior_par$param_dict$mu_prior_mean$get(),
                 mu_prior_info = py_main$prior_par$param_dict$mu_prior_info$get(),
                 tau_prior_alpha = py_main$prior_par$param_dict$tau_prior_alpha$get(),
                 tau_prior_beta = py_main$prior_par$param_dict$tau_prior_beta$get())

##############
# MCMC

stan_directory <- file.path(project_directory, "stan")
stan_model_name <- "logit_glmm"
model_file <- file.path(
  stan_directory, paste(stan_model_name, "stan", sep="."))
model_file_rdata <- file.path(
  stan_directory, paste(stan_model_name, "Rdata", sep="."))
# if (file.exists(model_file_rdata)) {
#   print("Loading pre-compiled Stan model.")
#   load(model_file_rdata)
# } else {
  # Run this to force re-compilation of the model.
  print("Compiling Stan model.")
  # In the stan directory run
  # $GIT_REPO_LOC/StanSensitivity/python/generate_models.py --base_model=logit_glmm.stan
  model_file <- file.path(stan_directory, paste(stan_model_name, "_generated.stan", sep=""))
  model <- stan_model(model_file)
  stan_sensitivity_model <- GetStanSensitivityModel(
    file.path(stan_directory, "logit_glmm"), stan_dat)
  save(model, stan_sensitivity_model, file=model_file_rdata)
# }


# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
seed <- 42
chains <- 1
cores <- 1 # Use one core for the sensitivity analysis.
iters <- 1000

# MCMC draws.
mcmc_time <- Sys.time()
stan_dat$mu_prior_epsilon <- 0
stan_sim <- sampling(
  model, data=stan_dat, seed=seed, iter=iters, chains=chains, cores=cores)
mcmc_time <- Sys.time() - mcmc_time

# ADVI.
advi_time <- Sys.time()
stan_advi <- vb(model, data=stan_dat,  algorithm="meanfield",
                output_samples=iters)
advi_time <- Sys.time() - advi_time

# Get a MAP estimate.
bfgs_map_time <- Sys.time()
stan_map_bfgs <- optimizing(
  model, data=stan_dat, algorithm="BFGS", hessian=TRUE,
  init=get_inits(stan_sim)[[1]], verbose=TRUE,
  tol_obj=1e-12, tol_grad=1e-12, tol_param=1e-12)
bfgs_map_time <- Sys.time() - bfgs_map_time

stan_map <- stan_map_bfgs
map_time <- bfgs_map_time


# Get the sensitivity results.
stopifnot(cores == 1) # rstansensitivity only supports one core for now.
draws_mat <- rstan::extract(stan_sim, permute=FALSE)[,1,]
mcmc_sens_time <- Sys.time()
sens_result <- GetStanSensitivityFromModelFit(stan_sim, draws_mat, stan_sensitivity_model)
mcmc_sens_time <- Sys.time() - mcmc_sens_time


# Save the results to an RData file for further post-processing.
stan_draws_file <- file.path(
  data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
save(stan_sim, mcmc_time, stan_dat,
     sens_result, stan_sensitivity_model, mcmc_sens_time,
     stan_advi, advi_time,
     stan_map, map_time,
     chains, cores,
     file=stan_draws_file)
print(stan_draws_file)
