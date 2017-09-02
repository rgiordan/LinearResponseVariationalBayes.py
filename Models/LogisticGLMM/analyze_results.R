library(rstan)
library(lme4)

library(dplyr)
library(reshape2)
library(trust)
library(tidyr)

library(LRVBUtils)

library(mvtnorm)
library(gridExtra)

library(jsonlite)

library(reticulate)
use_python("/usr/bin/python3")


project_directory <- file.path(
  Sys.getenv("GIT_REPO_LOC"),
  "LinearResponseVariationalBayes.py/Models/LogisticGLMM")
data_directory <- file.path(project_directory, "data/")

source(file.path(project_directory, "logit_glmm_lib.R"))
# source(file.path(project_directory, "densities_lib.R"))

analysis_name <- "criteo_subsampled"
#analysis_name <- "simulated_data_small"

# If true, save the results to a file readable by knitr.
save_results <- TRUE
results_file <- file.path(data_directory,
                          paste(analysis_name, "sensitivity.Rdata", sep="_"))

stan_draws_file <- file.path(
  data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
stan_results <- LoadIntoEnvironment(stan_draws_file)

glmer_file <- file.path(
  data_directory, paste(analysis_name, "_glmer_results.Rdata", sep=""))
glmer_results <- LoadIntoEnvironment(glmer_file)

# Load the python pickled data.
InitializePython()
py_main <- reticulate::import_main()
python_filename <- file.path(
  data_directory, paste(analysis_name, "_python_vb_results.pkl", sep=""))
reticulate::py_run_string(
"
pkl_file = open('" %_% python_filename %_% "', 'rb')
vb_results = pickle.load(pkl_file)
pkl_file.close()
")

vb_results <- py_main$vb_results

##############
# Check covariance

elbo_hess <- vb_results$elbo_hess
moment_jac <- vb_results$moment_jac

lrvb_cov <- vb_results$lrvb_cov
min(diag(lrvb_cov))
max(diag(lrvb_cov))

lrvb_sd_scale <- sqrt(diag(vb_results$lrvb_cov))
stopifnot(min(diag(vb_results$lrvb_cov)) > 0)

#################################
# Extract stuff from the python results

log_prior_hess <- t(vb_results$log_prior_hess)
vb_prior_sens <- vb_results$vb_prior_sens

glmm_par <- py_main$logit_glmm$get_glmm_parameters(
  K=stan_results$stan_dat$K, NG=stan_results$stan_dat$NG)
glmm_par$set_free(vb_results$glmm_par_free)
stopifnot(max(abs(glmm_par$get_vector() - vb_results$glmm_par_vector)) < 1e-8)
moment_par <- py_main$logit_glmm$MomentWrapper(glmm_par)
moment_par$set_moments(vb_results$glmm_par_free)


###################
# Make a dataframe out of the mean and covariance results.

draws_mat <- as.matrix(stan_results$stan_sim)

moment_par$set_moments(vb_results$glmm_par_free)
vb_mean_vec <- moment_par$moment_par$get_vector()

mfvb_sd_vec <- sqrt(GetMFVBCovVector(glmm_par))
mcmc_sd_vec <- apply(draws_mat, sd, MARGIN=2)
lrvb_sd_vec <- sqrt(diag(lrvb_cov))

# The MAP estimate
stan_map <- stan_results$stan_map 

# This is time-consuming and not necessary for the paper.
# Get the MAP standard deviation based on the negative inverse Hessian
# at the MAP, with the same names as the draws matrix.
# If we wanted the sds of log(tau), we'd need the delta method.
# For tidiness, let's just leave it as NA.
#inv_hess_diag <- -diag(solve(stan_map$hessian))
#names(inv_hess_diag) <- colnames(draws_mat)[1:length(inv_hess_diag)]


# Combine into a single tidy dataframe.
results_posterior <-
  rbind(
    ConvertPythonMomentVectorToDF(vb_mean_vec, glmm_par) %>% mutate(method="mfvb", metric="mean"),
    ConvertPythonMomentVectorToDF(mfvb_sd_vec, glmm_par) %>% mutate(method="mfvb", metric="sd"),
    ConvertPythonMomentVectorToDF(lrvb_sd_vec, glmm_par) %>% mutate(method="lrvb", metric="sd"),
    
    ConvertStanVectorToDF(colMeans(draws_mat), colnames(draws_mat), glmm_par) %>%
      mutate(method="mcmc", metric="mean"),
    ConvertStanVectorToDF(mcmc_sd_vec, colnames(draws_mat), glmm_par) %>%
      mutate(method="mcmc", metric="sd"),

    ConvertGlmerResultToDF(glmer_results$glmm_list, glmm_par) %>% mutate(method="glmer", metric="mean"),

    ConvertStanVectorToDF(stan_map$par, names(stan_map$par), glmm_par) %>%
      mutate(method="map", metric="mean")
    # ,
    # ConvertStanVectorToDF(sqrt(inv_hess_diag), names(inv_hess_diag$par), glmm_par) %>%
    #   mutate(method="map", metric="sd")
  )

results <- dcast(results_posterior, par + metric + component ~ method, value.var="val")


################################################
# Prior sensitivity.

# The Stan sensitivity and VB sensitivity from Python are stored in a different
# order.  We need to map one to the other, which we do using the prior_index_df dataframe.
prior_par <- py_main$logit_glmm$get_default_prior_params(K=stan_results$stan_dat$K)
prior_par$set_vector(array(1:prior_par$vector_size()))

prior_index_df <-
  RecursiveUnpackParameter(prior_par$dictval()) %>%
  rename(par=par_1, component_2=par_2, vb_index=val) %>%
  mutate(component_1=component, component_2=as.integer(component_2), vb_index=as.integer(vb_index))

matrix_ud_index <- function(i, j) {
  matrix_ud_index_ordered <- function(i, j) {
    as.integer((j - 1) + i * (i - 1) / 2 + 1)
  }
  ifelse(i > j, matrix_ud_index_ordered(i, j), matrix_ud_index_ordered(j, i))  
}

# Make the "component" column of matrix prior parameters into a linearized index
prior_index_df <- prior_index_df %>%
  mutate(component_ud=matrix_ud_index(component_1, component_2)) %>%
  mutate(component=case_when(!is.na(component_2) ~ matrix_ud_index(component_1, component_2),
                             TRUE ~ component))

# "Too few values" warnings are ok.
suppressWarnings(
  prior_index_df <-
    tibble(par=rownames(stan_results$sens_result$sens_mat),
           stan_index=1:nrow(stan_results$sens_result$sens_mat)) %>%
    separate(par, into=c("par", "component_1", "component_2"), sep="\\.") %>%
    mutate(component_1=as.integer(component_1), component_2=as.integer(component_2)) %>%
    inner_join(prior_index_df, by=c("component_1", "component_2", "par"))
)


### Extract the MCMC sensitivity.

mcmc_sens_mat <- stan_results$sens_result$sens_mat
mcmc_sens_list <- list()
for (prior_ind in unique(prior_index_df$stan_index)) {
  mcmc_sens_list[[length(mcmc_sens_list) + 1]] <-
    ConvertStanVectorToDF(mcmc_sens_mat[prior_ind, ], rownames(mcmc_sens_mat), glmm_par) %>%
      mutate(method="mcmc", metric="prior_sensitivity", stan_index=prior_ind)
}
mcmc_sens_df <- do.call(rbind, mcmc_sens_list) %>%
  inner_join(prior_index_df, by="stan_index", suffix=c("", "_prior"))


### Extract the VB sensitivity.

vb_sens_list <- list()
for (prior_ind in unique(prior_index_df$vb_index)) {
  vb_sens_list[[length(vb_sens_list) + 1]] <-
    ConvertPythonMomentVectorToDF(vb_prior_sens[, prior_ind], glmm_par) %>%
    mutate(method="lrvb", metric="prior_sensitivity", vb_index=prior_ind)
}
vb_sens_df <-
  do.call(rbind, vb_sens_list) %>%
  inner_join(prior_index_df, by="vb_index", suffix=c("", "_prior"))


# Get standard deviations for normalizing the sensitivities
results_sd <-
  filter(results_posterior, metric == "sd", method %in% c("lrvb", "mcmc")) %>%
  select(par, component, method, val) %>%
  rename(post_sd=val)

sens_df <-
  rbind(vb_sens_df, mcmc_sens_df) %>%
  select(-vb_index, -stan_index) %>%
  rename(component_1_prior=component_1, component_2_prior=component_2)

sens_df_norm <-
  inner_join(sens_df, results_sd, by=c("par", "component", "method")) %>%
  mutate(metric="prior_sensitivity_norm", val=val / post_sd) %>%
  select(-post_sd)

# Group together diagonal and off-diagonal elements of the sensitivity
# to the information matrix.  Note that NA values do not trip a case_when():
# > case_when(NA ~ 0, TRUE ~ 1)
# > 1
# Note that we can add up the normalized sensitivity of a single posterior component
# to several prior components because the denominator -- the posterior standard deviation --
# is the same for each prior component.
sensitivity_results <-
  rbind(sens_df, sens_df_norm) %>%
  mutate(diag_prior=case_when(is.na(component_2_prior) | is.na(component_1_prior) ~ FALSE,
                              TRUE ~ component_1_prior == component_2_prior)) %>%
  mutate(par_prior=case_when(diag_prior ~ paste(par_prior, "diag", sep="_"),
                             TRUE ~ par_prior)) %>%
  group_by(par, component, method, metric, par_prior) %>%
  summarize(val=sum(val), n=n())

unique(sensitivity_results[c("par_prior", "n")]) # Sanity check

sens_df_cast <-
  dcast(sensitivity_results,
        par + component + metric + par_prior ~ method,
        value.var="val")

if (save_results) {
  mcmc_time <- as.numeric(stan_results$mcmc_time, units="secs")
  map_time <- as.numeric(stan_results$map_time, units="secs")
  glmer_time <- as.numeric(glmer_results$glmm_list$glmm_time, units="secs")
  vb_time <- as.numeric(vb_results$vb_time, units="secs")
  hess_time <- as.numeric(vb_results$hess_time, units="secs")
  inverse_time <- as.numeric(vb_results$inverse_time, units="secs")
  num_mcmc_draws <- nrow(as.matrix(stan_results$stan_sim))
  num_gh_points <- vb_results$num_gh_points
  
  # Prior parameters
  pp <- stan_results$stan_dat
  pp$y_group <- NULL
  pp$y <- NULL
  pp$x <- NULL

  num_obs <- stan_results$stan_dat$N
  num_groups <- stan_results$stan_dat$NG
  beta_dim <- stan_results$stan_dat$K
  # Doesn't work with sparse elbo_hess
  #elbo_hess_sparsity <- Matrix(abs(vb_results$elbo_hess) > 1e-8)
  elbo_hess_sparsity <- 0.
  save(results,
       sens_df_cast,
       mcmc_time, glmer_time, map_time, vb_time, hess_time, inverse_time,
       num_mcmc_draws, num_gh_points,
       pp, num_obs, num_groups, beta_dim,
       elbo_hess_sparsity,
       file=results_file)
}


########################################
# Graphs and analysis

stop("Graphs follow -- not executing.")


if (FALSE) {
  # VB means:
  grid.arrange(
    ggplot(filter(results, metric == "mean", par != "e_u")) +
      geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
      geom_abline(aes(intercept=0, slope=1))
  ,   
    ggplot(filter(results, metric == "mean", par == "e_u")) +
      geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
      geom_abline(aes(intercept=0, slope=1))
  , ncol=2
  )

  # GLMER means:
  grid.arrange(
    ggplot(filter(results, metric == "mean", par != "e_u")) +
      geom_point(aes(x=mcmc, y=glmer, color=par), size=3) +
      geom_abline(aes(intercept=0, slope=1))
    ,   
    ggplot(filter(results, metric == "mean", par == "e_u")) +
      geom_point(aes(x=mcmc, y=glmer, color=par), size=3) +
      geom_abline(aes(intercept=0, slope=1)) +
      expand_limits(x=0, y=0)
    , ncol=2
  )

  # MAP means:  
  grid.arrange(
    ggplot(filter(results, metric == "mean", par != "e_u")) +
      geom_point(aes(x=mcmc, y=map, color=par), size=3) +
      geom_abline(aes(intercept=0, slope=1))
    ,   
    ggplot(filter(results, metric == "mean", par == "e_u")) +
      geom_point(aes(x=mcmc, y=map, color=par), size=3) +
      geom_abline(aes(intercept=0, slope=1))
    , ncol=2
  )
  
  # VB stdevs:
  grid.arrange(
    ggplot(filter(results, metric == "sd", par != "e_u")) +
      geom_point(aes(x=mcmc, y=mfvb, color="mfvb", shape=par), size=3) +
      geom_point(aes(x=mcmc, y=lrvb, color="lrvb", shape=par), size=3) +
      geom_abline(aes(intercept=0, slope=1))
  ,
    ggplot(filter(results, metric == "sd", par == "e_u")) +
      geom_point(aes(x=mcmc, y=mfvb, color="mfvb", shape=par), size=3) +
      geom_point(aes(x=mcmc, y=lrvb, color="lrvb", shape=par), size=3) +
      geom_abline(aes(intercept=0, slope=1))
  ,  
  ncol=2)

}


if (FALSE) {
  ggplot(filter(sens_df_cast, metric=="prior_sensitivity")) + 
    geom_point(aes(x=mcmc, y=lrvb, color=par_prior, shape=par), size=2) +
    geom_abline(aes(slope=1, intercept=0))
  
  ggplot(filter(sens_df_cast, metric=="prior_sensitivity_norm")) + 
    geom_point(aes(x=mcmc, y=lrvb, color=par_prior, shape=par), size=2) +
    geom_abline(aes(slope=1, intercept=0))
}


# Sanity check
if (FALSE) {
  foo <- rbind(sens_df, sens_df_norm) %>%
    dcast(par + component + par_prior + component_prior +
            component_1_prior + component_2_prior + method ~ metric,
          value.var="val")
  ggplot(foo) + 
    geom_point(aes(prior_sensitivity, prior_sensitivity_norm, color=par)) +
    geom_abline(aes(slope=1, intercept=0))
}





# Overall

# ggplot(
#   filter(results, metric == "mean") %>%
#     dcast(par + component + group ~ method, value.var="val") %>%
#     mutate(is_u = par == "u")) +
#   geom_point(aes(x=mcmc, y=mfvb, color=par), size=3) +
#   geom_abline(aes(intercept=0, slope=1)) +
#   facet_grid(~ is_u)
# 
# ggplot(
#   filter(results, metric == "sd") %>%
#     dcast(par + component + group ~ method, value.var="val") %>%
#     mutate(is_u = par == "u")) +
#   geom_point(aes(x=mcmc, y=mfvb, shape=par, color="mfvb"), size=3) +
#   geom_point(aes(x=mcmc, y=lrvb, shape=par, color="lrvb"), size=3) +
#   geom_abline(aes(intercept=0, slope=1)) +
#   facet_grid(~ is_u) +
#   ggtitle("Posterior standard deviations")
# 
# ggplot(
#   filter(results, metric == "sd", par != "u") %>%
#     dcast(par + component + group ~ method, value.var="val")
# ) +
#   geom_point(aes(x=mcmc, y=mfvb, color="mfvb", shape=par), size=3) +
#   geom_point(aes(x=mcmc, y=lrvb, color="lrvb", shape=par), size=3) +
#   expand_limits(x=0, y=0) +
#   xlab("MCMC (ground truth)") + ylab("VB") +
#   scale_color_discrete(guide=guide_legend(title="Method")) +
#   geom_abline(aes(intercept=0, slope=1))
# 
# ggplot(
#   filter(results, metric == "sd", par == "mu") %>%
#     dcast(par + component + group ~ method, value.var="val")
# ) +
#   geom_point(aes(x=mcmc, y=mfvb, color="mfvb", shape=par), size=3) +
#   geom_point(aes(x=mcmc, y=lrvb, color="lrvb", shape=par), size=3) +
#   expand_limits(x=0, y=0) +
#   xlab("MCMC (ground truth)") + ylab("VB") +
#   scale_color_discrete(guide=guide_legend(title="Method")) +
#   geom_abline(aes(intercept=0, slope=1))
# 
# 
# ggplot(
#   filter(results, metric == "sd", par != "u") %>%
#     dcast(par + component + group ~ method, value.var="val")
# ) +
#   geom_point(aes(x=mcmc, y=map, color="map", shape=par), size=3) +
#   expand_limits(x=0, y=0) +
#   xlab("MCMC (ground truth)") + ylab("MAP") +
#   scale_color_discrete(guide=guide_legend(title="Method")) +
#   geom_abline(aes(intercept=0, slope=1))
# 
# ggplot(
#   filter(results, metric == "mean") %>%
#     dcast(par + component + group ~ method, value.var="val") %>%
#     mutate(is_u = par == "u")) +
#   geom_point(aes(x=truth, y=mcmc, shape=par, color="mcmc"), size=3) +
#   geom_point(aes(x=truth, y=mfvb, shape=par, color="mfvb"), size=3) +
#   geom_point(aes(x=truth, y=map, shape=par, color="map"), size=3) +
#   geom_abline(aes(intercept=0, slope=1)) +
#   facet_grid(~ is_u)
# 
# 
# # Sensitivity
# 
# ggplot(filter(vb_prior_sens_cast, par != "u")) +
#   geom_point(aes(x=lrvb_norm, y=mcmc_norm, color=par)) +
#   geom_abline(aes(intercept=0, slope=1))
# 
# 
# # Note: mcmc_norm_sd is not here because it doens't work with the aggregation.
# 
# # Compare LRVB with the MCMC standard deviations
# ggplot(filter(vb_prior_sens_cast, par=="u")) +
#   geom_point(aes(x=lrvb_norm, y=mcmc_norm, color=prior_par)) +
#   geom_errorbar(aes(x=lrvb_norm,
#                     ymin=mcmc_norm - 2 * mcmc_norm_sd,
#                     ymax=mcmc_norm + 2 * mcmc_norm_sd,
#                     color=prior_par)) +
#   geom_abline(aes(intercept=0, slope=1))
# 
# # Compare MCMC with its own estimated standard deviations.
# ggplot(filter(vb_prior_sens_cast, par=="u")) +
#   geom_point(aes(x=mcmc_norm, y=mcmc_norm, color=prior_par)) +
#   geom_errorbar(aes(x=mcmc_norm,
#                     ymin=mcmc_norm - 2 * mcmc_norm_sd,
#                     ymax=mcmc_norm + 2 * mcmc_norm_sd,
#                     color=prior_par)) +
#   geom_abline(aes(intercept=0, slope=1))
# 
# ggplot(filter(vb_prior_sens_cast, par=="u")) +
#   geom_point(aes(x=mcmc_norm, y=mcmc_norm_small, color=prior_par)) +
#   geom_errorbar(aes(x=mcmc_norm,
#                     ymin=mcmc_norm_small - 2 * mcmc_norm_small_sd,
#                     ymax=mcmc_norm_small + 2 * mcmc_norm_small_sd,
#                     color=prior_par)) +
#   geom_abline(aes(intercept=0, slope=1))
# 
# ggplot(filter(vb_prior_sens_cast, par=="u")) +
#   geom_point(aes(x=mcmc, y=mcmc_small, color=prior_par)) +
#   geom_errorbar(aes(x=mcmc_norm,
#                     ymin=mcmc_small - 2 * mcmc_small_sd,
#                     ymax=mcmc_small + 2 * mcmc_small_sd,
#                     color=prior_par)) +
#   geom_abline(aes(intercept=0, slope=1))
