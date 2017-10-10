
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
  mutate(diff=val - val_truth, abs_diff=abs(diff), rel_diff = diff / val_truth) %>%
  filter(method != "bootstrap")

results_cast <-
  dcast(results_truth, par + component + metric + num_obs_per_group ~ method,
        value.var = "rel_diff") %>%
  mutate(n_scaling = 1 / sqrt(num_obs_per_group))


results_re_sd <- filter(results_cast, metric == "sd", par == "e_u")

ggplot(results_re_sd) + 
  geom_boxplot(aes(x=num_obs_per_group, y=mfvb, group=num_obs_per_group, color="mfvb")) +
  geom_boxplot(aes(x=num_obs_per_group, y=lrvb, group=num_obs_per_group, color="lrvb")) +
  geom_boxplot(aes(x=num_obs_per_group, y=jackknife, group=num_obs_per_group, color="jackknife")) +
  geom_hline(aes(yintercept=0))


ggplot(filter(results_truth, metric == "sd", par == "e_u")) +
  geom_boxplot(aes(x=num_obs_per_group, y=rel_diff, group=num_obs_per_group)) +
  facet_grid(~ method) +
  geom_hline(aes(yintercept=0)) +
  ggtitle("Relative error in standard deviation for random effects") +
  xlab("Number of observations per group") +
  ylab(TeX("$\\frac{\\sigma - \\sigma_{truth}}{\\sigma_{truth}}$"))


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

