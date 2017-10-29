# # # Plot of model training loss curves
library(ggplot2)

# load data
content_loss <- read.csv('run_neural_network-tag-content_loss.csv')
style_loss <- read.csv('run_neural_network-tag-style_loss.csv')
tv_loss <- read.csv('run_neural_network-tag-tv_loss.csv')
total_loss <- read.csv('run_neural_network-tag-total_loss.csv')

content_loss$log_loss <- log(content_loss$Value)
style_loss$log_loss <- log(style_loss$Value)
tv_loss$log_loss <- log(tv_loss$Value)
total_loss$log_loss <- log(total_loss$Value)

# format data
df_plot = list(
  content = content_loss,
  style = style_loss,
  tv = tv_loss,
  total = total_loss
)

# plot curve and save
for (loss_type in names(df_plot)) {
  df <- df_plot[loss_type][[1]]
  
  # plot loss curve
  fig_loss <- ggplot(data = df, aes(x = Step, y = Value)) + 
    geom_line() + theme_bw() + xlim(c(0, 1000)) + 
    labs(x = 'number of step', y = paste(loss_type, 'loss')) + 
    theme(panel.background = element_blank())
  ggsave(paste(loss_type, '_loss.pdf', sep = ''), fig_loss, width = 10, height = 8)
  
  # plot log loss curve
  fig_log_loss <- ggplot(data = df, aes(x = Step, y = log_loss)) + 
    geom_line() + theme_bw() + xlim(c(0, 1000)) + 
    labs(x = 'number of step', y = paste(loss_type, 'log loss')) + 
    theme(panel.background = element_blank())
  ggsave(paste(loss_type, '_log_loss.pdf', sep = ''), fig_log_loss, width = 10, height = 8)
}
