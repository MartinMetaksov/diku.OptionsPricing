require(plotly)

cuda_multi_gpu01_1 <- read.csv("cuda-multi-gpu01.1.csv")
cuda_option_gpu01_1 <- read.csv("cuda-option-gpu01.1.csv")

# add type column
cuda_multi_gpu01_1$type <- "multi"
cuda_option_gpu01_1$type <- "option"

# join tables
data <- rbind(cuda_multi_gpu01_1, cuda_option_gpu01_1)
data$type <- as.factor(data$type)

# add description column
data$desc <- paste(data$type, " ", data$precision, ", reg ", data$registers, ", v", data$version, ", block ", data$block, ", sort ", data$sort, sep = "")

# add exe column
data$exe <- paste(data$precision, " reg", data$registers, sep = "")

# convert time to seconds
data$kernel.time <- data$kernel.time / 1000000
data$total.time <- data$total.time / 1000000

# split data by file
files <- split(data, data$file)

# create dropdown buttons per file
buttons <- lapply(seq_along(files), function(i) {
  # ther are as many traces in a box plot as colors, make the ones for this file visible
  types.count = length(levels(data$type))
  visibility <- as.list(rep(F, types.count  * length(files)))
  e <- i * types.count
  s <- e - types.count + 1
  visibility[s : e] <- T
  
  list(method = "restyle",
       args = list("visible", visibility),
       label = names(files)[i])
})

# create an empty plot
p <- plot_ly()

# add add a box plot for each file
for (i in seq_along(files)) {
  p <- p %>% add_boxplot(data = files[[i]], y = ~exe, x = ~kernel.time, color = ~type,
                         orientation = "h", visible = i == 1)
}

# add the layout
p <- p %>%
  layout(
    boxmode = "group",
    hovermode = "closest",
    margin = list(l = 100),
    xaxis = list(title = "Kernel time (sec)"), 
    yaxis = list(title = "Version"),
    annotations = list(list(text = "File:", x = 0, y = 1.085, yref = "paper", align = "left", showarrow = F)),
    updatemenus = list(list(
      direction = "down",
      xanchor = "left",
      yanchor = "top",
      x = 0.05,
      y = 1.1,
      buttons = buttons)))

print(p)
