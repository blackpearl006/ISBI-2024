suppressMessages(library('nonlinearTseries'))

#mROI = c(1,2,3,4,5,6,7,8,9,10) 
#mROI = c(11,12,13,14,15,16,17,18,19,20) 
#mROI = c(21,22,23,24,25,26,27,28,29,30)
mROI = c(31,32,33,34)

csv.path = "/Users/ninad/Documents/_CBR/Scripts/Recurrence plots/CSV files/DM4_mci.csv"

mdir <- "/Users/ninad/Documents/_CBR/Data/ROI CSV files/default_mode_ts/mci"
mfiles <- list.files(mdir, pattern = ".csv", full.names = TRUE)

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

for (file_path in mfiles) {
  mdata <- read_csv_file(file_path)
  for (ROI in mROI) {
    time_series <- mdata[[ paste0('roi', ROI)]]
    emb_dim <- 0
    tau <- NULL
    tryCatch(
      {
        tau.acf <- timeLag(time_series, technique = "acf", selection.method = "first.minimum", lag.max = NULL, do.plot = FALSE)
        emb_dim <- estimateEmbeddingDim(time_series, time.lag = tau.acf, max.embedding.dim = 15)
        
        subject_name <- basename(file_path)
        results_df <- rbind(results_df, c(subject_name, ROI, emb_dim, tau.acf))
        
        cat("Done\n")
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
