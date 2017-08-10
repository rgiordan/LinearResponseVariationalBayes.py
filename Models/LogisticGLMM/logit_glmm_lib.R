# Helper functions for analyzing the Logit LRVB results in R.

library(reticulate)
library(purrr)
library(dplyr)

RecursiveUnpackParameter <- function(par, level=0, id_name="par") {
  if (is.numeric(par)) {
    if (length(par) == 1) {
      return(tibble(val=par))
    } else {
      return(tibble(val=par, component=1:length(par)))
    }
  } else if (is.list(par)) {
    next_level <- map(par, RecursiveUnpackParameter, level + 1)
    return(bind_rows(next_level, .id=paste(id_name, level + 1, sep="_")))
  }
}


InitializePython <- function(git_repo_loc=Sys.getenv("GIT_REPO_LOC")) {
  `%_%` <- function(x, y) { paste(x, y, sep="") }
  py_run_string("import sys")
  for (py_lib in c("LinearResponseVariationalBayes.py",
                   "LinearResponseVariationalBayes.py/Models",
                   "autograd")) {
    py_run_string("sys.path.append('" %_% file.path(git_repo_loc, py_lib) %_% "')")
  }
  py_run_string("import VariationalBayes as vb")
  py_run_string("import LogisticGLMM_lib as logit_glmm")
}


# Pack MCMC samples as returned by extract() and pack them into a matrix
# in the same order as the VB moments.
PackMCMCSamplesIntoMoments <- function(mcmc_sample, glmm_par, py_main=reticulate::import_main()) {
  mcmc_moment_par <- py_main$logit_glmm$MomentWrapper(glmm_par)
  mcmc_param_dict <- mcmc_moment_par$moment_par$param_dict
  
  num_draws <- dim(mcmc_sample$beta)[1]
  draws_mat <- matrix(NA, nrow=mcmc_moment_par$moment_par$vector_size(), ncol=num_draws)
  pb <- txtProgressBar(style=3, max=num_draws)
  for (draw in 1:num_draws) {
    setTxtProgressBar(pb, draw)
    beta <- array(mcmc_sample$beta[draw, ])
    mu <- mcmc_sample$mu[draw]
    tau <- mcmc_sample$tau[draw]
    u <- array(mcmc_sample$u[draw, ])
    
    mcmc_param_dict$e_beta$set(beta)
    mcmc_param_dict$e_beta_outer$set(beta %*% t(beta))
    mcmc_param_dict$e_mu$set(mu)
    mcmc_param_dict$e_mu2$set(mu^2)
    mcmc_param_dict$e_tau$set(tau)
    mcmc_param_dict$e_log_tau$set(log(tau))
    mcmc_param_dict$e_u$set(u)
    mcmc_param_dict$e_u2$set(u^2)
    draws_mat[, draw] <- mcmc_moment_par$moment_par$get_vector()
  }
  close(pb)
  
  return(draws_mat)
}

