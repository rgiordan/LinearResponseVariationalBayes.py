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

#analysis_name <- "criteo_subsampled"
#analysis_name <- "simulated_data_small"
analysis_name <- "simulated_data_for_refit"

# If true, save the results to a file readable by knitr.
save_results <- TRUE
results_file <- file.path(data_directory,
                          paste(analysis_name, "jackknife.Rdata", sep="_"))

InitializePython()
py_main <- reticulate::import_main()

# Original results
# python_filename <- file.path(
#   data_directory, paste(analysis_name, "_python_vb_results.pkl", sep=""))

# Results evaluating the jacknnife
python_jackknife_filename <- file.path(
  data_directory,
  paste(analysis_name, "_python_jackknife_results.pkl", sep=""))

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
kl_hess_chol = cholesky(kl_hess)
moment_jac_sens = kl_hess_chol.solve_A(moment_jac.T)
lrvb_cov = np.matmul(moment_jac, moment_jac_sens)
mfvb_mean = model.moment_wrapper.get_moment_vector_from_free(glmm_par_free)
")

# Get different standard deviations.
lrvb_cov <- py_main$lrvb_cov
lrvb_sd <- sqrt(diag(lrvb_cov))
mfvb_sd <- sqrt(GetMFVBCovVector(py_main$glmm_par))

reticulate::py_run_string("
moment_vec_samples = [ \
  model.moment_wrapper.get_moment_vector_from_free(opt_x) for opt_x in  glmm_par_sims ]
moment_vec_samples = np.array(moment_vec_samples)
")
moment_vec_samples <- py_main$moment_vec_samples
class(moment_vec_samples)
resample_sd <- apply(moment_vec_samples, MARGIN=2, sd)
resample_mean <- apply(moment_vec_samples, MARGIN=2, mean)

results <- rbind(
  ConvertPythonMomentVectorToDF(py_main$mfvb_mean, py_main$glmm_par) %>%
    mutate(method="mfvb", metric="mean"),
  ConvertPythonMomentVectorToDF(lrvb_sd, py_main$glmm_par) %>%
    mutate(method="lrvb", metric="sd"),
  ConvertPythonMomentVectorToDF(mfvb_sd, py_main$glmm_par) %>%
    mutate(method="mfvb", metric="sd"),
  ConvertPythonMomentVectorToDF(resample_sd, py_main$glmm_par) %>%
    mutate(method="truth", metric="sd"),
  ConvertPythonMomentVectorToDF(resample_mean, py_main$glmm_par) %>%
    mutate(method="truth", metric="mean")
)


results_cast <- dcast(results, par + component + metric ~ method, value.var = "val")
grid.arrange(
  ggplot(filter(results_cast, metric == "mean")) + 
    geom_point(aes(x=truth, y=mfvb, color=par)) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("Mean accuracy")
,
  ggplot(filter(results_cast, metric == "sd")) + 
    geom_point(aes(x=truth, y=lrvb, shape=par, color="lrvb"), size=3) +
    geom_point(aes(x=truth, y=mfvb, shape=par, color="mfvb"), size=3) +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle("SD accuracy")
, ncol=2)