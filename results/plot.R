#!/usr/bin/env Rscript
require(plotly)
require(htmlwidgets)
require(jsonlite)
require(dplyr)
require(plyr)
require(reshape2)
require(processx)

readFuthark <- function(file, real) {
  json <- fromJSON(file)
  datasets <- json[[1]][["datasets"]]
  
  df <- data.frame()
  for (i in 1:length(datasets)) {
    set <- datasets[[i]]
    name <- tools::file_path_sans_ext(basename(names(datasets)[i]))
    time <- min(set[["runtimes"]])
    df <- rbind(df, list(file=name, precision=real, total.time=time), stringsAsFactors=F)
  }
  df
}

title = "results"
datasets.folder = "raw"
datasets.names = c("CUDA-option", "CUDA-multi", "Futhark-basic", "Futhark-flat", "Sequential")
datasets.files = c("option.csv", "multi.csv", "basic32.json", "basic64.json", "flat32.json", "flat64.json", "seq.csv")

# load and join data
n <- 1
f <- 1
data <- data.frame()
while (n <= length(datasets.names)) {
  file <- file.path(datasets.folder, datasets.files[f])
  name <- datasets.names[n]
  
  if (endsWith(file, "csv")) {
    set <- read.csv(file, na.strings = "-")
    set$type <- name
    data <- bind_rows(data, set)
    
  } else if (endsWith(file, "json")) {
    set <- readFuthark(file, "float")
    set$type <- name
    data <- bind_rows(data, set)
    
    f <- f + 1
    file <- file.path(datasets.folder, datasets.files[f])
    set <- readFuthark(file, "double")
    set$type <- name
    data <- bind_rows(data, set)
  }
  
  f <- f + 1
  n <- n + 1
}

# sort precision
data <- data[order(data$precision),] 

# convert columns to factors
data$sort <- as.character(data$sort)
data$sort[is.na(data$sort)] <- "-"
data$sort <- as.factor(data$sort)
data$version <- as.factor(data$version)
data$block <- as.factor(data$block)
data$precision <- as.factor(data$precision)

# convert time to seconds
data$kernel.time <- data$kernel.time / 1000000
data$total.time <- data$total.time / 1000000

# convert memory to GB
data$memory <- data$memory / (1024 * 1024)

# create a scatter plot per file
plotScatter <- function() {
  
  sdata <- data.frame(data)
  
  # add exe column
  noreg <- is.na(sdata$registers)
  sdata$exe[noreg] <- as.character(sdata$type[noreg])
  sdata$exe[!noreg] <- paste(sdata$type[!noreg], " reg", sdata$registers[!noreg], sep = "")
  sdata$exe <- as.factor(sdata$exe)
  
  # add details column
  nodetails <- is.na(sdata$version)
  sdata$details[!nodetails] <- paste("version ", sdata$version[!nodetails], ", block ", sdata$block[!nodetails], ", sort ", sdata$sort[!nodetails], sep = "")
  
  # split data by file
  files <- split(sdata, sdata$file)
  
  # create dropdown buttons per file
  buttons <- lapply(seq_along(files), function(i) {
    # ther are as many traces in a box plot as colors, make the ones for this file visible
    colors.count = length(levels(data$precision))
    visibility <- as.list(rep(F, colors.count  * length(files)))
    e <- i * colors.count
    s <- e - colors.count + 1
    visibility[s : e] <- T
    
    list(method = "restyle",
         args = list("visible", visibility),
         label = names(files)[i])
  })
  
  # create an empty plot
  p <- plot_ly()
  
  # add add a box plot for each file
  for (i in seq_along(files)) {
    p <- add_trace(p, type = "scatter", data = files[[i]], x = ~exe, y = ~total.time, color = ~precision, visible = i == 1,
                   hoverinfo = "all", text = ~details)
  }
  
  # add the layout
  p <- p %>%
    layout(
      boxmode = "group",
      hovermode = "closest",
      margin = list(l = 100),
      xaxis = list(title = "Version"),
      yaxis = list(title = "Total time (sec)"), 
      annotations = list(list(text = "File:", x = 0, y = 1.085, yref = "paper", align = "left", showarrow = F)),
      updatemenus = list(list(
        direction = "down",
        xanchor = "left",
        yanchor = "top",
        x = 0.1,
        y = 1.1,
        buttons = buttons)))
  
  print(p)
  
  saveWidget(p, file = paste(title, "html", sep = ".")) 
}

mapSort <- function(d) {
  mapvalues(d, from=c("-","w","W", "h", "H"), to=c("None ", "Width ▲", "Width ▼", "Height ▲","Height ▼"))
}

mapFilesTitle <- function(d) {
  mapvalues(d, from=c("0_UNIFORM", "1_RAND", "2_RANDCONSTHEIGHT", "3_RANDCONSTWIDTH", "4_SKEWED", "5_SKEWEDCONSTHEIGHT", "6_SKEWEDCONSTWIDTH"),
            to=c("Dataset 0 - Uniform", "Dataset 1 - Random", "Dataset 2 - Random Constant Height", "Dataset 3 - Random Constant Width", "Dataset 4 - Skewed", "Dataset 5 - Skewed Constant Height", "Dataset 6 - Skewed Constant Width"))
}

mapFilesAxis <- function(d) {
  mapvalues(d, from=c("0_UNIFORM", "0_UNIFORM_1000", "0_UNIFORM_10000", "0_UNIFORM_30000", "1_RAND", "1_RAND_1000", "1_RAND_10000", "1_RAND_30000", "2_RANDCONSTHEIGHT", "2_RANDCONSTHEIGHT_1000", "2_RANDCONSTHEIGHT_10000", "2_RANDCONSTHEIGHT_30000", "3_RANDCONSTWIDTH", "3_RANDCONSTWIDTH_1000", "3_RANDCONSTWIDTH_10000", "3_RANDCONSTWIDTH_30000", "4_SKEWED", "5_SKEWEDCONSTHEIGHT", "6_SKEWEDCONSTWIDTH"),
            to=c("0 Uniform ", "0 Uniform  1k ", "0 Uniform 10k ", "0 Uniform 30k ", "1 Random ", "1 Random  1k ", "1 Random 10k ", "1 Random 30k ", "2 Random \nconst height ", "2 Random \nconst height  1k ", "2 Random \nconst height 10k ", "2 Random \nconst height 30k ", "3 Random \nconst width ", "3 Random \nconst width  1k ", "3 Random \nconst width 10k ", "3 Random \nconst width 30k ", "4 Skewed ", "5 Skewed \nconst height ", "6 Skewed \nconst width "))
}

mapFilesCSV <- function(d) {
  mapvalues(d, from=c("0_UNIFORM", "1_RAND", "2_RANDCONSTHEIGHT", "3_RANDCONSTWIDTH", "4_SKEWED", "5_SKEWEDCONSTHEIGHT", "6_SKEWEDCONSTWIDTH"),
            to=c("DAT 0", "DAT 1", "DAT 2", "DAT 3", "DAT 4", "DAT 5", "DAT 6"))
}

mapScale <- function(d, s) {
  round(s / d, 3)
}

makeBarPlot <- function(d, y, y_title, x_title, name, y_order = NA) {
  d$fileTitle <- mapFilesTitle(d$file)
  files <- split(d, d$file)

  for (i in seq_along(files)) {
    title <- names(files)[i]
    plotTitle <- files[[i]]$fileTitle
    p <-
      plot_ly(files[[i]]) %>%
      add_trace(x = ~float, y = y, type = 'bar', name = "Float", text = ~float, textposition = "auto",
                marker = list(color = 'rgb(239,138,98)',
                              line = list(color = 'rgb(8,48,107)', width = 0.7))) %>%
      add_trace(x = ~double, y = y, type = 'bar', name = "Double", text = ~double, textposition = "auto",
                marker = list(color = 'rgb(103,169,207)',
                              line = list(color = 'rgb(8,48,107)', width = 0.7))) %>%
      layout(title = plotTitle,
             barmode = 'group',
             autosize = F,
             margin = list(t = 120),
             font = list(family = "sans serif", size = 60),
             xaxis = list(title = x_title),
             yaxis = list(title = y_title, autorange = "reversed", categoryorder = "array", categoryarray = y_order),
             legend = list(orientation = 'h', x = -0.18))
    orca(p, paste(name, "-", title, ".png", sep = ""), width = 1400, height = 1300)
  }
}

makeBarPlotTypes <- function(d, title, name) {
  x_title <- "Speed-up (higher is better)"
  y_title <- "Dataset"
  colnames(d) <- sub("-", ".", colnames(d), fixed = TRUE)
  d$file <- mapFilesAxis(d$file)
  
  d$CUDA.option <- mapScale(d$CUDA.option, d$Sequential)
  d$CUDA.multi <- mapScale(d$CUDA.multi, d$Sequential)
  d$Futhark.basic <- mapScale(d$Futhark.basic, d$Sequential)
  d$Futhark.flat <- mapScale(d$Futhark.flat, d$Sequential)
  d$CUDA.option.text <- paste(d$CUDA.option)
  d$CUDA.multi.text <- paste(d$CUDA.multi)
  d$Futhark.basic.text <- paste(d$Futhark.basic)
  d$Futhark.flat.text <- paste(d$Futhark.flat)
  
  p <-
    plot_ly(d) %>%
    add_trace(x = ~CUDA.option, y = ~file, type = 'bar', name = "CUDA-option", text = ~CUDA.option.text, textposition = "auto",
              marker = list(color = 'rgb(102,194,165',
                            line = list(color = 'rgb(8,48,107)', width = 0.7))) %>%
    add_trace(x = ~CUDA.multi, y = ~file, type = 'bar', name = "CUDA-multi", text = ~CUDA.multi.text, textposition = "auto",
              marker = list(color = 'rgb(252,141,98)',
                            line = list(color = 'rgb(8,48,107)', width = 0.7))) %>%
    add_trace(x = ~Futhark.basic, y = ~file, type = 'bar', name = "Futhark-basic", text = ~Futhark.basic.text, textposition = "auto",
              marker = list(color = 'rgb(141,160,203)',
                            line = list(color = 'rgb(8,48,107)', width = 0.7))) %>%
    add_trace(x = ~Futhark.flat, y = ~file, type = 'bar', name = "Futhark-flat", text = ~Futhark.flat.text, textposition = "auto",
              marker = list(color = 'rgb(231,138,195)',
                            line = list(color = 'rgb(8,48,107)', width = 0.7))) %>%
    layout(title = title,
           barmode = 'group',
           margin = list(t = 120),
           font = list(family = "sans serif", size = 60),
           xaxis = list(title = x_title, ticksuffix = "x"),
           yaxis = list(title = y_title, autorange = "reversed"),
           legend = list(orientation = 'v'))
  orca(p, paste(name, ".png", sep = ""), width = 2300, height = 2800)
}

writeCSV <- function(float, double, name) {
  colnames(float) <- mapFilesCSV(colnames(float))
  colnames(double) <- mapFilesCSV(colnames(double))
  write.csv(float, file = paste(name, "-float.csv", sep = ""), row.names = F, quote=F)
  write.csv(double, file = paste(name, "-double.csv", sep = ""), row.names = F, quote=F)
}

# CUDA-option
plotCudaOptionBlocks <- function() {
  subset <- data[data$type == datasets.names[1], c(1,2,5,8)]
  cast <- dcast(subset, file + block ~ precision, value.var = "total.time", min)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, block ~ file, value.var = "float")
  double <- dcast(cast, block ~ file, value.var = "double")
  writeCSV(float, double, "option-blocks")
  # printing plots
  cast$block <- paste(cast$block, " ", sep = "")
  makeBarPlot(cast, ~block, "Block size", "Time (sec)", "option-blocks")
}

plotCudaOptionVersions <- function() {
  subset <- data[data$type == datasets.names[1], c(1,2,4,8)]
  cast <- dcast(subset, file + version ~ precision, value.var = "total.time", min)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, version ~ file, value.var = "float")
  double <- dcast(cast, version ~ file, value.var = "double")
  writeCSV(float, double, "option-versions")
  # printing plots
  cast$version <- paste(cast$version, " ", sep = "")
  makeBarPlot(cast, ~version, "Version", "Time (sec)", "option-versions")
}

plotCudaOptionSorts <- function() {
  subset <- data[data$type == datasets.names[1], c(1,2,6,8)]
  subset$sort <- mapSort(subset$sort)
  cast <- dcast(subset, file + sort ~ precision, value.var = "total.time", min)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, sort ~ file, value.var = "float")
  double <- dcast(cast, sort ~ file, value.var = "double")
  writeCSV(float, double, "option-sorts")
  # printing plots
  cast$sort <- paste(cast$sort, " ", sep = "")
  makeBarPlot(cast, ~sort, "Sorting", "Time (sec)", "option-sorts")
}

plotCudaOptionVersionsMem <- function() {
  subset <- data[data$type == datasets.names[1], c(1,2,4,9)]
  cast <- dcast(subset, file + version ~ precision, value.var = "memory", mean)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, version ~ file, value.var = "float")
  double <- dcast(cast, version ~ file, value.var = "double")
  writeCSV(float, double, "mem-option-versions")
  # printing plots
  cast$version <- paste(cast$version, " ", sep = "")
  makeBarPlot(cast, ~version, "Version", "Memory (MB)", "mem-option-versions")
}

printCudaOptionSorts <- function() {
  subset <- data[data$type == datasets.names[1], c(1,2,6,7,8)]
  subset$prep <- (subset$total.time - subset$kernel.time) * 1000
  subset <- subset[, c(1,2,3,6)]
  subset$sort <- mapSort(subset$sort)
  cast <- dcast(subset, file + sort ~ precision, value.var = "prep", mean)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, sort ~ file, value.var = "float")
  double <- dcast(cast, sort ~ file, value.var = "double")
  writeCSV(float, double, "option-sorts-prep")
}

# CUDA-multi
plotCudaMultiBlocks <- function() {
  subset <- data[data$type == datasets.names[2], c(1,2,5,8)]
  cast <- dcast(subset, file + block ~ precision, value.var = "total.time", min)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, block ~ file, value.var = "float")
  double <- dcast(cast, block ~ file, value.var = "double")
  writeCSV(float, double, "multi-blocks")
  # printing plots
  cast$block <- paste(cast$block, " ", sep = "")
  makeBarPlot(cast, ~block, "Block size", "Time (sec)", "multi-blocks")
}

plotCudaMultiVersions <- function() {
  subset <- data[data$type == datasets.names[2], c(1,2,4,8)]
  cast <- dcast(subset, file + version ~ precision, value.var = "total.time", min)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, version ~ file, value.var = "float")
  double <- dcast(cast, version ~ file, value.var = "double")
  writeCSV(float, double, "multi-versions")
  # printing plots
  cast$version <- paste(cast$version, " ", sep = "")
  makeBarPlot(cast, ~version, "Version", "Time (sec)", "multi-versions")
}

plotCudaMultiSorts <- function() {
  subset <- data[data$type == datasets.names[2], c(1,2,6,8)]
  subset$sort <- mapSort(subset$sort)
  cast <- dcast(subset, file + sort ~ precision, value.var = "total.time", min)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, sort ~ file, value.var = "float")
  double <- dcast(cast, sort ~ file, value.var = "double")
  writeCSV(float, double, "multi-sorts")
  # printing plots
  cast$sort <- paste(cast$sort, " ", sep = "")
  makeBarPlot(cast, ~sort, "Sorting", "Time (sec)", "multi-sorts")
}

plotCudaMultiVersionsMem <- function() {
  subset <- data[data$type == datasets.names[2], c(1,2,4,9)]
  cast <- dcast(subset, file + version ~ precision, value.var = "memory", mean)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, version ~ file, value.var = "float")
  double <- dcast(cast, version ~ file, value.var = "double")
  writeCSV(float, double, "mem-multi-versions")
  # printing plots
  cast$version <- paste(cast$version, " ", sep = "")
  makeBarPlot(cast, ~version, "Version", "Memory (MB)", "mem-multi-versions")
}

printCudaMultiVersions <- function() {
  subset <- data[data$type == datasets.names[2], c(1,2,4,7,8)]
  subset$prep <- (subset$total.time - subset$kernel.time) * 1000
  subset <- subset[, c(1,2,3,6)]
  cast <- dcast(subset, file + version ~ precision, value.var = "prep", mean)
  # printing tables
  cast$float <- round(cast$float, 3)
  cast$double <- round(cast$double, 3)
  float <- dcast(cast, version ~ file, value.var = "float")
  double <- dcast(cast, version ~ file, value.var = "double")
  writeCSV(float, double, "multi-versions-prep")
}

plotTypes <- function() {
  subset <- data[data$file == "0_UNIFORM_1000" | data$file ==  "0_UNIFORM_10000" | data$file == "0_UNIFORM_30000" | data$file == "1_RAND_1000" | data$file == "1_RAND_10000" | data$file == "1_RAND_30000", c(1,2,8,10)]
  cast <- dcast(subset, file + type ~ precision, value.var = "total.time", min)
  
  float <- dcast(cast[, -3], file ~ type)
  double <- dcast(cast[, -4], file ~ type)
  
  makeBarPlotTypes(float, "Comparison of Parallel Implementations (Float)", "all-approaches-float")
  makeBarPlotTypes(double, "Comparison of Parallel Implementations (Double)", "all-approaches-double")
  
  cast$type <- paste(cast$type, " ", sep = "")
  makeBarPlot(cast, ~type, "Parallel implementation", "Time (sec)", "all-approaches", datasets.names)
}

plotAll <- function() {
  setwd(paste(datasets.folder, "plots", sep = "-"))
  plotCudaOptionBlocks()
  plotCudaOptionVersions()
  plotCudaOptionSorts()
  plotCudaOptionVersionsMem()
  printCudaOptionSorts()
  
  plotCudaMultiBlocks()
  plotCudaMultiVersions()
  plotCudaMultiSorts()
  plotCudaMultiVersionsMem()
  printCudaMultiVersions()
  plotTypes()
  setwd("..")
}
