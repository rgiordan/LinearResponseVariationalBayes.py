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

analysis_name <- "simulated_data_small"

# If true, save the results to a file readable by knitr.
save_results <- TRUE
results_file <- file.path(data_directory,
                          paste(analysis_name, "jackknife_and_bootstrap.Rdata", sep="_"))

#################
# Load the python data

InitializePython()
py_main <- reticulate::import_main()

# Results evaluating the jacknnife
python_jackknife_filename <- file.path(
  data_directory,
  paste(analysis_name, "_python_vb_jackknife_results.pkl", sep=""))

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

# Load the data.
reticulate::py_run_string("
import os
analysis_name = '" %_% analysis_name %_% "'
data_dir = os.path.join(os.environ['GIT_REPO_LOC'],
                        'LinearResponseVariationalBayes.py/Models/LogisticGLMM/data')
json_filename = os.path.join(data_dir, '%s_stan_dat.json' % analysis_name)
y_g_vec, y_vec, x_mat, glmm_par, prior_par = logit_glmm.load_json_data(json_filename)
")

reticulate::py_run_string("
glmm_par_free = vb_jackknife_results['base_free_par']
beta_dim = x_mat.shape[1]
num_groups = np.max(y_g_vec) + 1
glmm_par = logit_glmm.get_glmm_parameters(K=beta_dim, NG=num_groups)
glmm_par.set_free(glmm_par_free)
prior_par = logit_glmm.get_default_prior_params(beta_dim)
moment_jac = vb_jackknife_results['moment_jac']
num_gh_points = vb_jackknife_results['num_gh_points']
model = logit_glmm.LogisticGLMM(
    glmm_par=glmm_par,
    prior_par=prior_par,
    x_mat=x_mat,
    y_vec=y_vec,
    y_g_vec=y_g_vec,
    num_gh_points=num_gh_points)
")

names(py_main$vb_jackknife_results)

reticulate::py_run_string("
kl_hess = obj_lib.unpack_csr_matrix(vb_jackknife_results['kl_hess'])
kl_hess_chol = cholesky(kl_hess)

weight_jacobian = obj_lib.unpack_csr_matrix(vb_jackknife_results['weight_jacobian'])
param_boot_mat = obj_lib.unpack_csr_matrix(vb_jackknife_results['param_boot_mat'])
moment_jac_sens = kl_hess_chol.solve_A(moment_jac.T)

lr_boot_free_par_list = vb_jackknife_results['lr_boot_free_par_list']
boot_free_par_list = vb_jackknife_results['boot_free_par_list']

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
lr_jackknife_moment_list = []
for obs in range(param_boot_mat.shape[1]):
  lr_diff = np.squeeze(np.asarray(param_boot_mat[:, obs].todense()))
  moment_vec = model.moment_wrapper.get_moment_vector_from_free(glmm_par_free - lr_diff)
  lr_jackknife_moment_list.append(moment_vec)

lr_jackknife_moments = np.array(lr_jackknife_moment_list)
")

lr_jackknife_moments <- py_main$lr_jackknife_moments
n_obs <- nrow(lr_jackknife_moments)
lr_jackknife_sd <- sqrt(n_obs - 1) * apply(lr_jackknife_moments, MARGIN=2, sd)
lr_jackknife_mean <- apply(lr_jackknife_moments, MARGIN=2, mean)


### Bootstrap

reticulate::py_run_string("
lr_bootstrap_moment_list = []
for free_par in lr_boot_free_par_list:
  moment_vec = model.moment_wrapper.get_moment_vector_from_free(free_par)
  lr_bootstrap_moment_list.append(moment_vec)
lr_bootstrap_moments = np.array(lr_bootstrap_moment_list)

bootstrap_moment_list = []
for free_par in boot_free_par_list:
  moment_vec = model.moment_wrapper.get_moment_vector_from_free(free_par)
  bootstrap_moment_list.append(moment_vec)
bootstrap_moments = np.array(bootstrap_moment_list)
")

lr_bootstrap_moments <- py_main$lr_bootstrap_moments
lr_bootstrap_sd <- apply(lr_bootstrap_moments, MARGIN=2, sd)
lr_bootstrap_mean <- apply(lr_bootstrap_moments, MARGIN=2, mean)

bootstrap_moments <- py_main$bootstrap_moments
bootstrap_sd <- apply(bootstrap_moments, MARGIN=2, sd)
bootstrap_mean <- apply(bootstrap_moments, MARGIN=2, mean)


### Look at bootstrap distributions.
GetSampleRowDataframe <- function(moment_vec, sample, method, metric) {
  ConvertPythonMomentVectorToDF(moment_vec, py_main$glmm_par) %>%
    mutate(method=method, metric=metric, sample=sample)
}

GetSampleDataframe <- function(moment_vec_array, method, metric) {
  df_list <- lapply(1:nrow(bootstrap_moments),
                     function(rownum) {
                       GetSampleRowDataframe(moment_vec_array[rownum, ], sample=rownum,
                                             method=method, metric=metric)
                      })
  return(do.call(rbind, df_list))
}

bootstrap_samples <-
  rbind(
    GetSampleDataframe(bootstrap_moments, method="bootstrap", metric="value"),
    GetSampleDataframe(lr_bootstrap_moments, method="lr_bootstrap", metric="value"))

ggplot(filter(bootstrap_samples, par == "e_beta")) + 
  geom_histogram(aes(x=val), bins=50) +
  facet_grid(method ~ component, scales="free")



### Combine and inspect

results <- rbind(
  ConvertPythonMomentVectorToDF(py_main$mfvb_mean, py_main$glmm_par) %>%
    mutate(method="mfvb", metric="mean"),
  ConvertPythonMomentVectorToDF(lrvb_sd, py_main$glmm_par) %>%
    mutate(method="lrvb", metric="sd"),
  ConvertPythonMomentVectorToDF(mfvb_sd, py_main$glmm_par) %>%
    mutate(method="mfvb", metric="sd"),

    ConvertPythonMomentVectorToDF(lr_jackknife_mean, py_main$glmm_par) %>%
    mutate(method="lr_jackknife", metric="mean"),
  ConvertPythonMomentVectorToDF(lr_jackknife_sd, py_main$glmm_par) %>%
    mutate(method="lr_jackknife", metric="sd"),
  
  ConvertPythonMomentVectorToDF(bootstrap_mean, py_main$glmm_par) %>%
    mutate(method="bootstrap", metric="mean"),
  ConvertPythonMomentVectorToDF(bootstrap_sd, py_main$glmm_par) %>%
    mutate(method="bootstrap", metric="sd"),

  ConvertPythonMomentVectorToDF(lr_bootstrap_mean, py_main$glmm_par) %>%
    mutate(method="lr_bootstrap", metric="mean"),
  ConvertPythonMomentVectorToDF(lr_bootstrap_sd, py_main$glmm_par) %>%
    mutate(method="lr_bootstrap", metric="sd"),
  
  stan_results
)


results_cast <-
  dcast(results, par + component + metric ~ method, value.var = "val") %>%
  mutate(is_re=par == "e_u")

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
  ggplot(filter(results_cast, metric == "mean", !is_re)) + 
    geom_point(aes(x=bootstrap, y=mfvb, shape=par, color="mfvb"), size=3) +
    geom_point(aes(x=bootstrap, y=lr_bootstrap, shape=par, color="lr_bootstrap"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("Mean accuracy")
,
  ggplot(filter(results_cast, metric == "sd", !is_re)) + 
    geom_point(aes(x=bootstrap, y=lrvb, shape=par, color="lrvb"), size=3) +
    geom_point(aes(x=bootstrap, y=mfvb, shape=par, color="mfvb"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("SD accuracy based on Hessian")
,
  ggplot(filter(results_cast, metric == "sd", !is_re)) + 
    geom_point(aes(x=bootstrap, y=lr_jackknife, shape=par, color="linear jackknife"), size=3) +
    geom_point(aes(x=bootstrap, y=lr_bootstrap, shape=par, color="linear bootstrap"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("SD accuracy based on linearization")
, ncol=3)


grid.arrange(
  ggplot(filter(results_cast, metric == "mean", is_re)) + 
    geom_point(aes(x=bootstrap, y=mfvb, shape=par, color="mfvb"), size=3) +
    geom_point(aes(x=bootstrap, y=lr_bootstrap, shape=par, color="lr_bootstrap"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("Mean accuracy")
  ,
  ggplot(filter(results_cast, metric == "sd", is_re)) + 
    geom_point(aes(x=bootstrap, y=lrvb, shape=par, color="lrvb"), size=3) +
    geom_point(aes(x=bootstrap, y=mfvb, shape=par, color="mfvb"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("SD accuracy based on Hessian")
  ,
  ggplot(filter(results_cast, metric == "sd", is_re)) + 
    geom_point(aes(x=bootstrap, y=lr_jackknife, shape=par, color="linear jackknife"), size=3) +
    geom_point(aes(x=bootstrap, y=lr_bootstrap, shape=par, color="linear bootstrap"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("SD accuracy based on linearization")
  , ncol=3)