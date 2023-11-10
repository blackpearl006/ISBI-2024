suppressMessages(library('nonlinearTseries'))

hROI = c(1,2,3,4,5,6,7,8,9,10) 
#hROI = c(11,12,13,14,15,16,17,18,19,20) 
#hROI = c(21,22,23,24,25,26,27,28,29,30)
#hROI = c(31,32,33,34)

csv.path = "/Users/ninad/Documents/_CBR/Scripts/Recurrence plots/CSV files/DM1_healthy_MI.csv"

hdir <- "/Users/ninad/Documents/_CBR/Data/ROI CSV files/default_mode_ts/healthy"
hfiles <- list.files(hdir, pattern = ".csv", full.names = TRUE)

read_csv_file <- function(file_path) {
  file_name <- basename(file_path)
  time_series <- read.csv(file_path, header = FALSE)
  numeric_data_list <- list()
  roi_names <- character()
  
  for (roi in seq(1, nrow(time_series))) {
    numeric_data <- as.numeric(time_series[roi,])
    roi_name <- paste0('roi', roi)
    numeric_data_list[[roi_name]] <- as.numeric(numeric_data)
    roi_names <- c(roi_names, roi_name)
  }
  
  names(numeric_data_list) <- roi_names
  return(numeric_data_list)
}

results_df <- data.frame()

for (file_path in hfiles) {
  hdata <- read_csv_file(file_path)
  for (ROI in hROI) {
    # Extract time series data for the current ROI
    time_series <- hdata[[ROI]]
    
    # Initialize variables for embedding dimension and time lag
    emb_dim <- 0
    tau <- NULL
    
    # Try to estimate embedding dimension and time lag
    tryCatch(
      {
        tau.acf <- timeLag(time_series, technique = "ami", selection.method = "first.minimum", lag.max = NULL, do.plot = FALSE)
        emb_dim <- estimateEmbeddingDim(time_series, time.lag = tau.acf, max.embedding.dim = 15, do.plot = FALSE)
        subject_name <- basename(file_path)
        results_df <- rbind(results_df, c(subject_name, ROI, emb_dim, tau.acf))
        
        cat("Done\n")
      },
      error = function(e) {
        # Handle errors, store 0 for embedding dimension and NA for time lag
        subject_name <- basename(file_path)
        results_df <- rbind(results_df, c(subject_name, ROI, 0, NA))
        cat("Error:", conditionMessage(e), "\n")
      }
    )
  }
}

colnames(results_df) <- c("subject", "ROI", "DIM", "Tau")
write.csv(results_df, file = csv.path, row.names = FALSE)