library(jsonlite)

filename <- "/tmp/foo.json"

my_df <- data.frame(a=runif(10), b=runif(10))
mat <- matrix(1:9, 3, 3)
foo_list <- list(a=runif(5), b=1, df=my_df, mat=mat)
foo_json <- toJSON(foo_list)
json_file <- file(filename, "w")
write(foo_json, file=json_file)
close(json_file)

# Note the dataframe can't be read by pandas it seems.
json_file <- file(filename, "r")
my_dat <- fromJSON(readLines(json_file, warn=FALSE))
close(json_file)
my_dat$df
my_dat$mat
mat[1, 3]


# Works ok
json_file <- file("/tmp/foo_python.json", "r")
json_dat <- fromJSON(readLines(json_file))
close(json_file)
json_dat
