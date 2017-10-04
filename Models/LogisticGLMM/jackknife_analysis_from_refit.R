library(lme4)
library(dplyr)
library(ggplot2)
library(reshape2)
library(tidyr)
library(LRVBUtils)
library(gridExtra)
library(reticulate)
use_python("/usr/bin/python3")

project_directory <- file.path(
  Sys.getenv("GIT_REPO_LOC"),
  "LinearResponseVariationalBayes.py/Models/LogisticGLMM")
data_directory <- file.path(project_directory, "data/")

source(file.path(project_directory, "logit_glmm_lib.R"))

analysis_name <- "simulated_data_for_refit"

# If true, save the results to a file readable by knitr.
save_results <- TRUE
results_file <- file.path(data_directory,
                          paste(analysis_name, "jackknife.Rdata", sep="_"))

#################
# Load the python data

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
glmm_par_sims = vb_jackknife_results['glmm_par_sims']
true_params = vb_jackknife_results['true_params']
glmm_par_free = vb_jackknife_results['glmm_par_free']
glmm_par = logit_glmm.get_glmm_parameters(K=true_params.beta_dim, NG=true_params.num_groups)
glmm_par.set_free(glmm_par_free)
prior_par = logit_glmm.get_default_prior_params(true_params.beta_dim)
moment_jac = vb_jackknife_results['moment_jac']
model = logit_glmm.LogisticGLMM(
    glmm_par=glmm_par,
    prior_par=prior_par,
    x_mat=vb_jackknife_results['x_mat'],
    y_vec=vb_jackknife_results['y_vec'],
    y_g_vec=vb_jackknife_results['y_g_vec'],
    num_gh_points=vb_jackknife_results['num_gh_points'])
")


reticulate::py_run_string("
kl_hess = obj_lib.unpack_csr_matrix(vb_jackknife_results['kl_hess'])
weight_jac = obj_lib.unpack_csr_matrix(vb_jackknife_results['weight_jac'])
kl_hess_chol = cholesky(kl_hess)
moment_jac_sens = kl_hess_chol.solve_A(moment_jac.T)
lrvb_cov = np.matmul(moment_jac, moment_jac_sens)
mfvb_mean = model.moment_wrapper.get_moment_vector_from_free(glmm_par_free)
")

##################
# Load the Stan data

stan_draws_file <- file.path(
  data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
stan_results <- LoadIntoEnvironment(stan_draws_file)
draws_mat <- as.matrix(stan_results$stan_sim)

mcmc_sd_vec <- apply(draws_mat, sd, MARGIN=2)

stan_results <- rbind(
  ConvertStanVectorToDF(colMeans(draws_mat), colnames(draws_mat), py_main$glmm_par) %>%
    mutate(method="mcmc", metric="mean"),
  ConvertStanVectorToDF(mcmc_sd_vec, colnames(draws_mat), py_main$glmm_par) %>%
    mutate(method="mcmc", metric="sd")
)


########################################
# Get different standard deviations.

### LRVB

lrvb_cov <- py_main$lrvb_cov
lrvb_sd <- sqrt(diag(lrvb_cov))
mfvb_sd <- sqrt(GetMFVBCovVector(py_main$glmm_par))

### Jackknife

reticulate::py_run_string("
param_boot_mat = kl_hess_chol.solve_A(weight_jac.T)
")

reticulate::py_run_string("
def get_jackknife_moments(obs):
  lr_diff = np.squeeze(np.asarray(param_boot_mat[:, obs].todense()))
  return model.moment_wrapper.get_moment_vector_from_free(glmm_par_free - lr_diff)

moment_par_jackknife_list = []
num_obs = model.x_mat.shape[0]
for obs in range(num_obs):
  if obs % 5000 == 0:
    print('Obs {} of {}'.format(obs, num_obs))
  moment_par_jackknife_list.append(get_jackknife_moments(obs))
moment_par_jacknnife = np.array(moment_par_jackknife_list)
")

moment_par_jacknnife <- py_main$moment_par_jacknnife
n_obs <- nrow(moment_par_jacknnife)
jackknife_sd <- sqrt(n_obs - 1) * apply(moment_par_jacknnife, MARGIN=2, sd)
jackknife_mean <- apply(moment_par_jacknnife, MARGIN=2, mean)



### Infinitesimal bootstrap

reticulate::py_run_string("
num_obs = model.x_mat.shape[0]
def get_bootstrap_moments():
  weight_diff = np.random.multinomial(num_obs, [1. / num_obs] * num_obs, size=1) - 1.0
  lr_diff = param_boot_mat * np.squeeze(np.asarray(weight_diff))
  print(lr_diff.shape)
  return model.moment_wrapper.get_moment_vector_from_free(glmm_par_free - lr_diff)

moment_par_bootstrap_list = []
num_bootstraps = 200
for boot_sample in range(num_bootstraps):
  moment_par_bootstrap_list.append(get_bootstrap_moments())
moment_par_bootstrap = np.array(moment_par_bootstrap_list)
")

moment_par_bootstrap <- py_main$moment_par_bootstrap
bootstrap_sd <- apply(moment_par_bootstrap, MARGIN=2, sd)
bootstrap_mean <- apply(moment_par_bootstrap, MARGIN=2, mean)



### Truth

reticulate::py_run_string("
moment_vec_samples = [ \
  model.moment_wrapper.get_moment_vector_from_free(opt_x) for opt_x in  glmm_par_sims ]
moment_vec_samples = np.array(moment_vec_samples)
")
moment_vec_samples <- py_main$moment_vec_samples
class(moment_vec_samples)
resample_sd <- apply(moment_vec_samples, MARGIN=2, sd)
resample_mean <- apply(moment_vec_samples, MARGIN=2, mean)


### Combine and inspect

results <- rbind(
  ConvertPythonMomentVectorToDF(py_main$mfvb_mean, py_main$glmm_par) %>%
    mutate(method="mfvb", metric="mean"),
  ConvertPythonMomentVectorToDF(lrvb_sd, py_main$glmm_par) %>%
    mutate(method="lrvb", metric="sd"),
  ConvertPythonMomentVectorToDF(mfvb_sd, py_main$glmm_par) %>%
    mutate(method="mfvb", metric="sd"),
  ConvertPythonMomentVectorToDF(jackknife_mean, py_main$glmm_par) %>%
    mutate(method="jackknife", metric="mean"),
  ConvertPythonMomentVectorToDF(jackknife_sd, py_main$glmm_par) %>%
    mutate(method="jackknife", metric="sd"),
  ConvertPythonMomentVectorToDF(bootstrap_mean, py_main$glmm_par) %>%
    mutate(method="bootstrap", metric="mean"),
  ConvertPythonMomentVectorToDF(bootstrap_sd, py_main$glmm_par) %>%
    mutate(method="bootstrap", metric="sd"),
  ConvertPythonMomentVectorToDF(resample_sd, py_main$glmm_par) %>%
    mutate(method="truth", metric="sd"),
  ConvertPythonMomentVectorToDF(resample_mean, py_main$glmm_par) %>%
    mutate(method="truth", metric="mean"),
  stan_results
)


results_cast <- dcast(results, par + component + metric ~ method, value.var = "val")

grid.arrange(
  ggplot(filter(results_cast, metric == "mean")) + 
    geom_point(aes(x=mcmc, y=mfvb, shape=par, color="vb")) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("Mean accuracy")
,
  ggplot(filter(results_cast, metric == "sd")) + 
    geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb")) +
    geom_point(aes(x=mcmc, y=lrvb, shape=par, color="lrvb")) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("SD accuracy")
, ncol=2
)

grid.arrange(
  ggplot(filter(results_cast, metric == "mean")) + 
    geom_point(aes(x=truth, y=mfvb, shape=par, color="vb")) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("Mean accuracy")
,
  ggplot(filter(results_cast, metric == "sd")) + 
    geom_point(aes(x=truth, y=lrvb, shape=par, color="lrvb"), size=3) +
    geom_point(aes(x=truth, y=mfvb, shape=par, color="mfvb"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("SD accuracy based on Hessian")
,
  ggplot(filter(results_cast, metric == "sd")) + 
    geom_point(aes(x=truth, y=jackknife, shape=par, color="jackknife"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("SD accuracy based on infinitesimal jackknife")
, ncol=3)