library(dplyr)
library(ggplot2)
library(reshape2)
library(jsonlite)

library(lme4)
library(rstan)

criteo_dir <- file.path(
    Sys.getenv("GIT_REPO_LOC"), "criteo/criteo_conversion_logs/")
clean_data_filename <- file.path(criteo_dir, "data_clean.Rdata")

load(clean_data_filename)

sample_groups <- unique(d_clean$V11)[1:100]
d_sub <- d_clean[d_clean$V11 %in% sample_groups, ]
d_sub$c <- 1
nrow(d_sub)

analysis_name <- "criteo_subsampled"

########################
# Run stan

project_directory <- file.path(
    Sys.getenv("GIT_REPO_LOC"),
    "LinearResponseVariationalBayes.py/Models/LogisticGLMM_R_code")
data_directory <- file.path(project_directory, "data/")

y <- as.integer(d_sub$conversion)
regressors <- paste("V", c(4, 5, 7, 9, 10), sep="")
x <- as.matrix(d_sub[regressors])
y_g_orig <- factor(d_sub$V11)
y_g <- as.integer(y_g_orig) - 1

k_reg <- ncol(x)

stan_dat <- list(NG = max(y_g) + 1,
                 N = length(y),
                 K = ncol(x),
                 y_group = y_g,
                 y = y,
                 x = x,
                 # Priors
                 beta_prior_mean = rep(0, k_reg),
                 beta_prior_var = 10. * diag(k_reg),
                 mu_prior_mean = 0.0,
                 mu_prior_var = 100.,
                 mu_prior_mean_c = 0.0,
                 mu_prior_var_c = 200.,
                 mu_prior_t = 1,
                 mu_prior_epsilon = 0,
                 tau_prior_alpha = 3.0,
                 tau_prior_beta = 3.0)

##############
# frequentist glmm

glmer_time <- Sys.time()
glmm_res <- glmer(conversion ~ V4 + V5 + V7 + V9 + V10 + (1|V11),
                  data=data.frame(d_sub), family="binomial", verbose=FALSE)
glmer_time <- Sys.time() - glmer_time

glmm_summary <- summary(glmm_res)
u <- data.frame(ranef(glmm_res)$V11)
names(u) <- "u"
u$V11 <- rownames(u)

u_df <- inner_join(u, d_sub[, "V11"], by="V11")
mean(u_df$u)
mean(u$u[y_g + 1])

glmm_list <- list()
glmm_list$beta_mean <- glmm_summary$coefficients[regressors, "Estimate"]
glmm_list$beta_par <- rownames(glmm_summary$coefficients[regressors, ])
glmm_list$mu_mean <- glmm_summary$coefficients["(Intercept)", "Estimate"]
glmm_list$mu_sd <- attr(glmm_summary$varcor$V11, "stddev")
glmm_list$u_map <- as.numeric(ranef(glmm_res)$V11[, "(Intercept)"])
glmm_list$glmm_time <- as.numeric(glmer_time, units="secs")



##############
# Export the data for fitting in Python.

json_filename <- file.path(
    data_directory, paste(analysis_name, "_stan_dat.json", sep=""))
json_file <- file(json_filename, "w")
json_list <- toJSON(list(stan_dat=stan_dat, glmm_fit=glmm_list))
write(json_list, file=json_file)
close(json_file)


##############
# MCMC


stan_directory <- file.path(project_directory, "stan")
stan_model_name <- "logit_glmm"
model_file <- file.path(
    stan_directory, paste(stan_model_name, "stan", sep="."))
model_file_rdata <- file.path(
    stan_directory, paste(stan_model_name, "Rdata", sep="."))
if (file.exists(model_file_rdata)) {
  print("Loading pre-compiled Stan model.")
  load(model_file_rdata)
} else {
  # Run this to force re-compilation of the model.
  print("Compiling Stan model.")
  model_file <- file.path(
      stan_directory, paste(stan_model_name, "stan", sep="."))
  model <- stan_model(model_file)
  save(model, file=model_file_rdata)
}


# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
seed <- 42
chains <- 1
cores <- 4

iters <- 20000

# Draw the draws and save.
mcmc_time <- Sys.time()
stan_dat$mu_prior_epsilon <- 0
stan_sim <- sampling(
    model, data=stan_dat, seed=seed, iter=iters, chains=chains, cores=cores)
mcmc_time <- Sys.time() - mcmc_time

# Sample with advi
advi_time <- Sys.time()
stan_advi <- vb(model, data=stan_dat,  algorithm="meanfield",
                output_samples=iters)
advi_time <- Sys.time() - advi_time

# Get a MAP estimate
bfgs_map_time <- Sys.time()
stan_map_bfgs <- optimizing(
    model, data=stan_dat, algorithm="BFGS", hessian=TRUE,
    init=get_inits(stan_sim)[[1]], verbose=TRUE,
    tol_obj=1e-12, tol_grad=1e-12, tol_param=1e-12)
bfgs_map_time <- bfgs_map_time - Sys.time()

stan_map <- stan_map_bfgs
map_time <- bfgs_map_time

# Save the fit to an RData file.
stan_draws_file <- file.path(
    data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
save(stan_sim, mcmc_time, stan_dat,
     stan_advi, advi_time,
     stan_map, map_time,
     chains, cores,
     file=stan_draws_file)
