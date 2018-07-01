#!/usr/bin/env Rscript
require(plotly)
require(htmlwidgets)

args = commandArgs(trailingOnly = TRUE)

# args = c("args", "results-wmp", "Futhark basic", "wmp-basic.csv", "Futhark flat", "wmp-flat.csv", "Cuda option", "wmp-option.csv", "Cuda multi", "wmp-multi.csv")

# load and join data
title = args[2]
i <- 3
data = data.frame()
while (i < length(args)) {
  a <- read.csv(args[i + 1], na.strings = "-")
  a$type <- args[i]
  data = rbind(data, a)
  i <- i + 2
}

# sort precision
data <- data[order(data$precision),] 

# convert type to factor
data$type <- as.factor(data$type)

# add exe column
noreg <- is.na(data$registers)
data$exe[noreg] <- as.character(data$type[noreg])
data$exe[!noreg] <- paste(data$type[!noreg], " reg", data$registers[!noreg], sep = "")
data$exe <- as.factor(data$exe)

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
  p <- p %>% add_boxplot(data = files[[i]], x = ~exe, y = ~total.time, color = ~precision, visible = i == 1)
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

# print(p)

saveWidget(p, file = paste(title, "html", sep = "."))
