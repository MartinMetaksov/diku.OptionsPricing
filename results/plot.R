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

# add exe column
noreg <- is.na(data$registers)
data$exe[noreg] <- as.character(data$type[noreg])
data$exe[!noreg] <- paste(data$type[!noreg], " reg", data$registers[!noreg], sep = "")
data$exe <- as.factor(data$exe)

# add details column
nodetails <- is.na(data$version)
data$details[!nodetails] <- paste("version ", data$version[!nodetails], ", block ", data$block[!nodetails], ", sort ", data$sort[!nodetails], sep = "")

# convert time to seconds
data$kernel.time <- data$kernel.time / 1000000
data$total.time <- data$total.time / 1000000

# split data by file
files <- split(data, data$file)

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

