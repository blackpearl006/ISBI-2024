suppressMessages(library('nonlinearTseries'))
network_dict <- list(
  DMN = 34,
  FP = 21,
  CO = 32,
  SM = 33,
  OP = 22,
  CB = 18
)

Subject_class <- 'CN'
for (Network in names(network_dict)) {
  hROI <- 1:network_dict[[Network]]
  csv.path <- sprintf("path/to/output/directory/DimTau_ADNI/%s_%s.csv", Network, Subject_class)
  hdir <- sprintf("path/to /dataset/ADNIFMRtimeseries/%s/%s", Subject_class, Network)
  hfiles <- list.files(hdir, pattern = ".csv", full.names = TRUE)
  read_csv_file <- function(file_path) {
    time_series <- read.csv(file_path, header = FALSE)
    numeric_data_list <- lapply(1:nrow(time_series), function(roi) as.numeric(time_series[roi, ]))
    names(numeric_data_list) <- paste0('roi', 1:nrow(time_series))
    return(numeric_data_list)
  }
  results_df <- data.frame()
  for (file_path in hfiles) {
    hdata <- read_csv_file(file_path)
    
    for (ROI in hROI) {
      time_series <- hdata[[ROI]]
      emb_dim <- 0
      tau <- NULL
      tryCatch(
        {
          tau.acf <- timeLag(time_series, technique = "acf", selection.method = "first.minimum", lag.max = NULL, do.plot = FALSE)
          emb_dim <- estimateEmbeddingDim(time_series, time.lag = tau.acf, max.embedding.dim = 15, do.plot = FALSE)
          subject_name <- basename(file_path)
          results_df <- rbind(results_df, c(subject_name, ROI, emb_dim, tau.acf))
        },
        error = function(e) {
          subject_name <- basename(file_path)
          results_df <- rbind(results_df, c(subject_name, ROI, 0, NA))
          cat("Error:", conditionMessage(e), "\n")
        }
      )
    }
  }
  colnames(results_df) <- c("subject", "ROI", "DIM", "Tau")
  write.csv(results_df, file = csv.path, row.names = FALSE)
}
cat("Processing complete for all networks.\n")
