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

InitializePython()
py_main <- reticulate::import_main()

read_analysis_results <- function(analysis_name) {
  #################
  # Load the python data

  # Results evaluating the jacknnife
  #python_jackknife_filename = os.path.join(data_dir, '%s_python_vb_jackknife_results.pkl' % analysis_name)
  #python_simulation_filename = os.path.join(data_dir, '%s_python_refit_jackknife_results.pkl' % analysis_name)
  
  python_jackknife_filename <-
    file.path(data_directory, sprintf('%s_python_vb_jackknife_results.pkl', analysis_name))
  python_simulation_filename <-
    file.path(data_directory, sprintf("%s_python_refit_jackknife_results.pkl", analysis_name))
  
  reticulate::py_run_string("
import VariationalBayes.SparseObjectives as obj_lib
from scikits.sparse.cholmod import cholesky
import numpy as np

pkl_file = open('" %_% python_jackknife_filename %_% "', 'rb')
vb_jackknife_results = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('" %_% python_simulation_filename %_% "', 'rb')
vb_sim_results = pickle.load(pkl_file)
pkl_file.close()
")
  
  names(py_main$vb_sim_results)
  names(py_main$vb_jackknife_results)
  
  reticulate::py_run_string("
glmm_par_sims = vb_sim_results['glmm_par_sims']
true_params = vb_sim_results['true_params']
model = logit_glmm.load_model_from_pickle(vb_sim_results)
moment_jac = vb_sim_results['moment_jac']
kl_hess = obj_lib.unpack_csr_matrix(vb_sim_results['kl_hess_packed'])
weight_jac = obj_lib.unpack_csr_matrix(vb_sim_results['weight_jac'])
kl_hess_chol = cholesky(kl_hess)
moment_jac_sens = kl_hess_chol.solve_A(moment_jac.T)
lrvb_cov = np.matmul(moment_jac, moment_jac_sens)
mfvb_mean = model.moment_wrapper.get_moment_vector_from_free(model.glmm_par.get_free())
param_boot_mat = kl_hess_chol.solve_A(weight_jac.T)

lr_boot_moment_vec_list_long = np.array(vb_jackknife_results['lr_boot_moment_vec_list_long'])
boot_moment_vec_list = np.array(vb_jackknife_results['boot_moment_vec_list'])

glmm_par = model.glmm_par
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
  
  ### Infinitesimal and full bootstrap
  
  lr_boot_moment_vec_list_long <- py_main$lr_boot_moment_vec_list_long
  lr_bootstrap_sd <- apply(lr_boot_moment_vec_list_long, MARGIN=2, sd)
  lr_bootstrap_mean <- apply(lr_boot_moment_vec_list_long, MARGIN=2, mean)
  
  full_moment_par_bootstrap <- py_main$boot_moment_vec_list
  full_bootstrap_sd <- apply(moment_par_bootstrap, MARGIN=2, sd)
  full_bootstrap_mean <- apply(moment_par_bootstrap, MARGIN=2, mean)
  
  ### Truth
  
  reticulate::py_run_string("
moment_vec_samples = [ \
  model.moment_wrapper.get_moment_vector_from_free(opt_x) for opt_x in glmm_par_sims ]
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
    
    ConvertPythonMomentVectorToDF(lr_bootstrap_mean, py_main$glmm_par) %>%
      mutate(method="bootstrap", metric="mean"),
    ConvertPythonMomentVectorToDF(lr_bootstrap_sd, py_main$glmm_par) %>%
      mutate(method="bootstrap", metric="sd"),
    
    ConvertPythonMomentVectorToDF(full_bootstrap_mean, py_main$glmm_par) %>%
      mutate(method="bootstrap", metric="mean"),
    ConvertPythonMomentVectorToDF(full_bootstrap_sd, py_main$glmm_par) %>%
      mutate(method="bootstrap", metric="sd"),
    
    ConvertPythonMomentVectorToDF(resample_sd, py_main$glmm_par) %>%
      mutate(method="truth", metric="sd"),
    ConvertPythonMomentVectorToDF(resample_mean, py_main$glmm_par) %>%
      mutate(method="truth", metric="mean"),
    stan_results
  ) %>%
    mutate(analysis_name=analysis_name, num_obs_per_group=num_obs_per_group)

  return(results)
}


# Debugging
num_obs_per_group = 2
analysis_name <- sprintf("simulated_data_for_refit_%d", num_obs_per_group)
results <- read_analysis_results(analysis_name)


## Graphs
if (FALSE) {
  
# 2, 5, 20, 40, 60, 100
results_list <- list()
for (num_obs_per_group in c(2, 5, 20, 40, 60, 100)) {
  print(num_obs_per_group)
  results_list[[length(results_list) + 1]] <- read_analysis_results(num_obs_per_group)
}
results <- do.call(rbind, results_list)

# If true, save the results to a file readable by knitr.
results_file <- file.path(data_directory, "jackknife_summary.Rdata")
save(results, file=results_file)

}



