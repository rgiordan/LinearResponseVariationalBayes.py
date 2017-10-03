library(lme4)
library(dplyr)
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
analysis_name <- "simulated_data_small"

# If true, save the results to a file readable by knitr.
save_results <- TRUE
results_file <- file.path(data_directory,
                          paste(analysis_name, "jackknife.Rdata", sep="_"))

InitializePython()
py_main <- reticulate::import_main()

# Original results
python_filename <- file.path(
  data_directory, paste(analysis_name, "_python_vb_results.pkl", sep=""))

# Results evaluating the jacknnife
python_jackknife_filename <- file.path(
  data_directory,
  paste(analysis_name, "_python_vb_jackknife_results.pkl", sep=""))

reticulate::py_run_string(
  "
pkl_file = open('" %_% python_filename %_% "', 'rb')
vb_results = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('" %_% python_jackknife_filename %_% "', 'rb')
vb_pert_results = pickle.load(pkl_file)
pkl_file.close()
")
