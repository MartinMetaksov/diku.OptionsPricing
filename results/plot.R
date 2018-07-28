#!/usr/bin/env Rscript
require(plotly)
require(htmlwidgets)
require(jsonlite)
require(dplyr)

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
datasets.names = c("Cuda option", "Cuda multi", "Futhark basic", "Futhark flat")
datasets.files = c("option.csv", "multi.csv", "basic32.json", "basic64.json", "flat32.json", "flat64.json")

# load and join data
n <- 1
f <- 1
data <- data.frame()
while (n <= length(datasets.names)) {
  file <- datasets.files[f]
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
    set <- readFuthark(datasets.files[f], "double")
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
data$type <- as.factor(data$type)
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

makeBarPlot <- function(d, x, x_title, y_title, name) {
  d$float <- round(d$float, 3)
  d$double <- round(d$double, 3)
  files <- split(d, d$file)
  for (i in seq_along(files)) {
    title <- names(files)[i]
    p <-
      plot_ly(files[[i]]) %>%
      add_trace(x = x, y = ~float, type = 'bar', name = "float", text = ~float, textposition = "auto",
                marker = list(color = 'rgb(239,138,98)',
                              line = list(color = 'rgb(8,48,107)', width = 0.7))) %>%
      add_trace(x = x, y = ~double, type = 'bar', name = "double", text = ~double, textposition = "auto",
                marker = list(color = 'rgb(103,169,207)',
                              line = list(color = 'rgb(8,48,107)', width = 0.7))) %>%
      add_annotations(
        yref="paper", 
        xref="paper", 
        y=1.08, 
        x=0.5, 
        text=title, 
        showarrow=F, 
        font=list(size=30)
      ) %>%
      layout(title = FALSE,
             barmode = 'group',
             font = list(family = "sans serif", size = 25),
             xaxis = list(title = x_title),
             yaxis = list(title = y_title),
             legend = list(orientation = 'h'))
    orca(p, paste(name, "-", title, ".png", sep = ""), scale = 2)
  }
}

# CUDA-option
plotCudaOptionBlocks <- function() {
  subset <- data[data$type == "Cuda option", c(1,2,5,8)]
  cast <- dcast(subset, file + block ~ precision, value.var = "total.time", min)
  makeBarPlot(cast, ~block, "Block size", "Time (sec)", "option-blocks")
}

plotCudaOptionVersions <- function() {
  subset <- data[data$type == "Cuda option", c(1,2,4,8)]
  cast <- dcast(subset, file + version ~ precision, value.var = "total.time", min)
  makeBarPlot(cast, ~version, "Version", "Time (sec)", "option-versions")
}

plotCudaOptionSorts <- function() {
  subset <- data[data$type == "Cuda option", c(1,2,6,8)]
  subset$sort <- mapvalues(subset$sort, 
                                 from=c("-","w","W", "h", "H"), 
                                 to=c("None", "Width asc.", "Width desc.", "Height asc.","Height desc."))
  cast <- dcast(subset, file + sort ~ precision, value.var = "total.time", min)
  makeBarPlot(cast, ~sort, "Sorting", "Time (sec)", "option-sorts")
}

plotCudaOptionVersionsMem <- function() {
  subset <- data[data$type == "Cuda option", c(1,2,4,9)]
  cast <- dcast(subset, file + version ~ precision, value.var = "memory", min)
  makeBarPlot(cast, ~version, "Version", "Memory (MB)", "mem-option-versions")
}

# CUDA-multi
plotCudaMultiVersions <- function() {
  subset <- data[data$type == "Cuda multi", c(1,2,4,8)]
  cast <- dcast(subset, file + version ~ precision, value.var = "total.time", min)
  makeBarPlot(cast, ~version, "Version", "Time (sec)", "multi-versions")
}

plotCudaMultiSorts <- function() {
  subset <- data[data$type == "Cuda multi", c(1,2,6,8)]
  subset$sort <- mapvalues(subset$sort, 
                           from=c("-","w","W", "h", "H"), 
                           to=c("None", "Width asc.", "Width desc.", "Height asc.","Height desc."))
  cast <- dcast(subset, file + sort ~ precision, value.var = "total.time", min)
  makeBarPlot(cast, ~sort, "Sorting", "Time (sec)", "multi-sorts")
}

plotCudaMultiVersionsMem <- function() {
  subset <- data[data$type == "Cuda multi", c(1,2,4,9)]
  cast <- dcast(subset, file + version ~ precision, value.var = "memory", min)
  makeBarPlot(cast, ~version, "Version", "Memory (MB)", "mem-multi-versions")
}

plotAll <- function() {
  setwd("plots")
  plotCudaOptionBlocks()
  plotCudaOptionVersions()
  plotCudaOptionSorts()
  plotCudaOptionVersionsMem()
  plotCudaMultiVersions()
  plotCudaMultiSorts()
  plotCudaMultiVersionsMem()
  setwd("..")
}
