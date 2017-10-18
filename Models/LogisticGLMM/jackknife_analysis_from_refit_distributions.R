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

# num_obs_per_group <- 40
# analysis_name <- sprintf("simulated_data_for_refit_%d", num_obs_per_group)
analysis_name <- sprintf("criteo_subsampled")


#################
# Load the python data

# Results evaluating the jacknnife
python_jackknife_filename <-
  file.path(data_directory, sprintf('%s_python_vb_jackknife_results.pkl', analysis_name))
# python_simulation_filename <-
#   file.path(data_directory, sprintf("%s_python_refit_jackknife_results.pkl", analysis_name))
python_simulation_filename <-
  file.path(data_directory, sprintf("%s_python_vb_results.pkl", analysis_name))

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
#glmm_par_sims = vb_sim_results['glmm_par_sims']
#true_params = vb_sim_results['true_params']
model = logit_glmm.load_model_from_pickle(vb_sim_results)
#moment_jac = vb_sim_results['moment_jac']
#kl_hess = obj_lib.unpack_csr_matrix(vb_sim_results['kl_hess_packed'])
#weight_jac = obj_lib.unpack_csr_matrix(vb_sim_results['weight_jac'])
#kl_hess_chol = cholesky(kl_hess)
#moment_jac_sens = kl_hess_chol.solve_A(moment_jac.T)
#lrvb_cov = np.matmul(moment_jac, moment_jac_sens)
#mfvb_mean = model.moment_wrapper.get_moment_vector_from_free(model.glmm_par.get_free())
#param_boot_mat = kl_hess_chol.solve_A(weight_jac.T)

lr_boot_moment_vec_list_long = np.array(vb_jackknife_results['lr_boot_moment_vec_list_long'])
boot_moment_vec_list = np.array(vb_jackknife_results['boot_moment_vec_list'])

glmm_par = model.glmm_par
")


### Infinitesimal and full bootstrap

lr_boot_moment_vec_list_long <- py_main$lr_boot_moment_vec_list_long
lr_bootstrap_sd <- apply(lr_boot_moment_vec_list_long, MARGIN=2, sd)
lr_bootstrap_mean <- apply(lr_boot_moment_vec_list_long, MARGIN=2, mean)

boot_moment_vec_list <- py_main$boot_moment_vec_list
full_bootstrap_sd <- apply(boot_moment_vec_list, MARGIN=2, sd)
full_bootstrap_mean <- apply(boot_moment_vec_list, MARGIN=2, mean)

# ### Truth
# reticulate::py_run_string("
# moment_vec_samples = [ \
# model.moment_wrapper.get_moment_vector_from_free(opt_x) for opt_x in glmm_par_sims ]
# moment_vec_samples = np.array(moment_vec_samples)
# ")
# 
# moment_vec_samples <- py_main$moment_vec_samples
# class(moment_vec_samples)
# resample_sd <- apply(moment_vec_samples, MARGIN=2, sd)
# resample_mean <- apply(moment_vec_samples, MARGIN=2, mean)

### Combine and inspect

ArrayToMomentDF <- function(moment_vec_array) {
  results_list <- list()
  for (ind in 1:nrow(moment_vec_array)) {
    results_list[[length(results_list) + 1]] <-
      ConvertPythonMomentVectorToDF(moment_vec_array[ind, ], py_main$glmm_par) %>%
      mutate(row=ind)
  }
  return(do.call(rbind, results_list))
}

boot_results <- ArrayToMomentDF(boot_moment_vec_list) %>%
  mutate(method="bootstrap", metric="draw")
linear_boot_results <- ArrayToMomentDF(lr_boot_moment_vec_list_long[1:500, ]) %>%
  mutate(method="linear_bootstrap", metric="draw")
# truth_results <- ArrayToMomentDF(moment_vec_samples) %>%
#   mutate(method="truth", metric="draw")

# results <- rbind(boot_results, linear_boot_results, truth_results)
results <- rbind(boot_results, linear_boot_results)

ggplot(filter(results, par == "e_u", component == 1)) +
  geom_histogram(aes(x=val, y=..density.., group=method),
                 bins=15, fill="black", color="salmon", lwd=2) +
  facet_grid(~ method)

names(py_main$vb_sim_results)
names(py_main$vb_jackknife_results$timer$time_dict)
py_main$vb_jackknife_results$timer$time_dict$weight_jac_time
py_main$vb_jackknife_results$timer$time_dict$hess_time
py_main$vb_jackknife_results$timer$time_dict$moment_jac_time

py_main$vb_sim_results$num_groups
nrow(py_main$vb_sim_results$x_mat)
py_main$vb_jackknife_results$timer$time_dict$bootstrap_time / 60
py_main$vb_jackknife_results$timer$time_dict$lr_bootstrap_time
nrow(boot_moment_vec_list)
nrow(lr_boot_moment_vec_list_long)



reticulate::py_run_string("
kl_hess = obj_lib.unpack_csr_matrix(vb_sim_results['kl_hess_packed'])
kl_sub_hess = kl_hess[0:200, 0:200].todense()
kl_hess_dim = kl_hess.shape
")

kl_hess <- Matrix(py_main$kl_sub_hess)
image(kl_hess != 0)
