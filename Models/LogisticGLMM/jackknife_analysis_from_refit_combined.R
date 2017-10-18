
library(lme4)
library(dplyr)
library(ggplot2)
library(reshape2)
library(tidyr)
library(LRVBUtils)
library(gridExtra)
library(reticulate)
library(latex2exp)
use_python("/usr/bin/python3")

project_directory <- file.path(
  Sys.getenv("GIT_REPO_LOC"),
  "LinearResponseVariationalBayes.py/Models/LogisticGLMM")
data_directory <- file.path(project_directory, "data/")

source(file.path(project_directory, "logit_glmm_lib.R"))

results_file <- file.path(data_directory, "jackknife_summary.Rdata")
load(results_file)


results_truth <- 
  inner_join(filter(results, method == "truth") %>% select(-method),
             filter(results, method != "truth"),
             by=c("par", "component", "metric", "num_obs_per_group", "analysis_name"),
             suffix=c("_truth", "")) %>%
  select(-analysis_name) %>%
  mutate(diff=val - val_truth, abs_diff=abs(diff), rel_diff = diff / val_truth)

results_cast <-
  dcast(results_truth, par + component + metric + num_obs_per_group ~ method,
        value.var = "rel_diff") %>%
  mutate(n_scaling = 1 / sqrt(num_obs_per_group))


results_re_sd <- filter(results_cast, metric == "sd", par == "e_u")

method_labels <- list()
method_labels["bootstrap"] <- "Full bootstrap"
method_labels["lrvb"] <- "Hessian"
method_labels["lr_bootstrap"] <- "Infinitesimal jackknife"
method_labeller <- function(variable, value) {
  if (value %in% names(method_labels)) {
    return(method_labels[value])
  } else {
    return(value)
  }
}

ggplot(filter(results_truth, metric == "sd", par == "e_u",
              method %in% c("bootstrap", "lrvb", "lr_bootstrap"))) +
  geom_boxplot(aes(x=num_obs_per_group, y=rel_diff, group=num_obs_per_group)) +
  facet_grid(~ method, labeller=method_labeller) +
  geom_hline(aes(yintercept=0)) +
  ggtitle("Relative error in standard deviation for random effects") +
  xlab("Number of observations per group") +
  ylab(TeX("$\\frac{\\sigma - \\sigma_{truth}}{\\sigma_{truth}}$"))


boot_cast <-
  dcast(filter(results, metric == "mean"),
        par + component + metric + num_obs_per_group ~ method,
        value.var = "val")

ggplot(filter(boot_cast, num_obs_per_group %in% c(2, 5, 20), par == "e_u")) +
  geom_point(aes(x=bootstrap - truth, y=lr_bootstrap - truth)) +
  facet_grid(~ num_obs_per_group,
             labeller=function(variable, value) { paste("Obs per group = ", value) }) +
  geom_abline(aes(slope=1, intercept=0)) +
  xlab("Full bootstrap change") + ylab("Infinitesimal jackknife change") +
  ggtitle("Evaluating the accuracy of the infinitesimal jackknife")
