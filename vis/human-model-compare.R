library(tidyverse)
library(ggtext)
library(ggrepel)

plot_df <- tribble(
  ~phenomenon,            ~stage, ~model,          ~accuracy, ~sd,
  
  # ---------------- Human ----------------
  "binding",              "10M",  "Human",         57.5, NA,
  "binding",              "50M",  "Human",         89.4, NA,
  "binding",              "100M", "Human",         87.3, NA,
  
  "adjunct island",       "10M",  "Human",         67.0, NA,
  "adjunct island",       "50M",  "Human",         78.0, NA,
  "adjunct island",       "100M", "Human",         94.0, NA,
  
  "complex NP island",    "10M",  "Human",         55.3, NA,
  "complex NP island",    "50M",  "Human",         92.5, NA,
  "complex NP island",    "100M", "Human",         80.0, NA,
  
  "question formation",   "10M",  "Human",         38.0, NA,
  "question formation",   "50M",  "Human",         80.0, NA,
  
  # ---------------- Transformer ----------------
  "binding",              "10M",  "Transformer",   53.4, 4.7,
  "binding",              "30M",  "Transformer",   57.3, 2.7,
  "binding",              "50M",  "Transformer",   62.1, 3.0,
  "binding",              "100M", "Transformer",   73.1, 1.7,
  
  "adjunct island",       "10M",  "Transformer",   69.8, 7.2,
  "adjunct island",       "30M",  "Transformer",   85.9, 3.7,
  "adjunct island",       "50M",  "Transformer",   91.7, 1.5,
  "adjunct island",       "100M", "Transformer",   94.3, 1.4,
  
  "complex NP island",    "10M",  "Transformer",   43.3, 9.4,
  "complex NP island",    "30M",  "Transformer",   60.5, 4.6,
  "complex NP island",    "50M",  "Transformer",   51.0, 11.3,
  "complex NP island",    "100M", "Transformer",   65.4, 8.4,
  
  "question formation",   "10M",  "Transformer",   55.0, 2.6,
  "question formation",   "30M",  "Transformer",   53.3, 0.9,
  "question formation",   "50M",  "Transformer",   54.1, 0.7,
  "question formation",   "100M", "Transformer",   51.9, 0.9,
  
  # ---------------- LSTM ----------------
  "binding",              "10M",  "LSTM",          51.2, 1.9,
  "binding",              "30M",  "LSTM",          51.7, 9.5,
  "binding",              "50M",  "LSTM",          59.7, 2.0,
  "binding",              "100M", "LSTM",          75.9, 1.7,
  
  "adjunct island",       "10M",  "LSTM",          89.3, 2.2,
  "adjunct island",       "30M",  "LSTM",          83.5, 4.9,
  "adjunct island",       "50M",  "LSTM",          90.1, 3.2,
  "adjunct island",       "100M", "LSTM",          94.6, 1.8,
  
  "complex NP island",    "10M",  "LSTM",          53.7, 7.6,
  "complex NP island",    "30M",  "LSTM",          53.5, 4.9,
  "complex NP island",    "50M",  "LSTM",          55.4, 4.2,
  "complex NP island",    "100M", "LSTM",          62.1, 3.8,
  
  "question formation",   "10M",  "LSTM",          53.5, 1.9,
  "question formation",   "30M",  "LSTM",          50.3, 1.7,
  "question formation",   "50M",  "LSTM",          48.2, 1.6,
  "question formation",   "100M", "LSTM",          45.1, 0.7
)



COLORS <- c(
  Human       = "#F89217",
  Transformer = "#023FA5",
  LSTM        = "#009E73"
)

FILLS <- c(
  Transformer = "#9BBEF8",
  LSTM        = "#A8DDB5"
)

SHAPES <- c(
  Human       = 16,
  Transformer = 15,
  LSTM        = 17
)


ann_labels <- tribble(
  ~phenomenon,          ~ref_label,                    ~age10,
  ~age50,                ~age100,
  
  "binding",
  "Chien & Wexler (1990)",
  "age: 4–4;6",
  "age: 6–6;6",
  "Adults",
  
  "adjunct island",
  "Goodluck et al. (1992)",
  "age: 3",
  "age: 4",
  "Adults",
  
  "complex NP island",
  "de Villiers & Roeper (1995)",
  "age: 3–4",
  "age: 4–5",
  "Adults",
  
  "question formation",
  "Crain & Nakayama (1987)",
  "age: 3;2–4;7",
  "age: 4;7–5;11",
  NA
)

plot_df <- left_join(
  plot_df,
  ann_labels,
  by = "phenomenon"
)



plot_df$stage <- factor(
  plot_df$stage,
  levels = c("10M", "30M", "50M", "100M")
)

plot_df$phenomenon <- factor(
  plot_df$phenomenon,
  levels = c(
    "binding",
    "adjunct island",
    "complex NP island",
    "question formation"
  )
)



age_df <- plot_df |>
  filter(model == "Human") |>
  mutate(
    age = case_when(
      stage == "10M"  ~ age10,
      stage == "50M"  ~ age50,
      stage == "100M" ~ age100,
      TRUE ~ NA_character_
    )
  ) |>
  filter(!is.na(age))



paper_df <- distinct(
  plot_df,
  phenomenon,
  ref_label
)


p <- ggplot(
  plot_df,
  aes(
    stage,
    accuracy,
    colour = model,
    group = model
  )
) +
  

geom_hline(
  yintercept = 50,
  colour = "grey65",
  linetype = "dashed",
  linewidth = 0.5
) +

geom_ribbon(
  data = filter(plot_df, model != "Human"),
  aes(
    x = stage,
    ymin = accuracy - sd,
    ymax = accuracy + sd,
    fill = model,
    group = model
  ),
  inherit.aes = FALSE,
  alpha = 0.15,
  linewidth = 0,
  colour = NA
) +


geom_line(
  data = filter(plot_df, model != "Human"),
  aes(
    stage,
    accuracy,
    colour = model,
    group = model
  ),
  linewidth = 1.15
) +
  

geom_point(
  data = filter(plot_df, model == "Human"),
  size = 3.5
) +
  

geom_point(
  aes(shape = model),
  size = 2.8
) +
  

geom_text(
  data = paper_df,
  aes(
    x = -Inf,
    y = Inf,
    label = ref_label
  ),
  inherit.aes = FALSE,
  hjust = -0.05,
  vjust = 1.4,
  fontface = "italic",
  colour = "grey35",
  size = 3.2
) +
geom_text_repel(
  data = age_df,
  aes(
    x = stage,
    y = accuracy,
    label = age
  ),
  inherit.aes = FALSE,
  
  colour = "#F89217",
  fontface = "bold",
  size = 3.0,
  
  direction = "both",
  min.segment.length = 0,
  segment.colour = "grey55",
  segment.linewidth = 0.5,
  box.padding = 0.5,
  point.padding = 0.8,
  force = 2,
  force_pull = 0.5,
  
  max.overlaps = Inf,
  seed = 123
) +
facet_wrap(
  ~phenomenon,
  ncol = 2
) +
coord_cartesian(
  ylim = c(0, 100),
  clip = "off"
) +
scale_colour_manual(
  values = COLORS
) +
  
  scale_fill_manual(
    values = FILLS
  ) +
  
  scale_shape_manual(
    values = SHAPES
  ) +
  
  scale_x_discrete(
    labels = c(
      "10M"  = "10M",
      "30M"  = "30M",
      "50M"  = "50M",
      "100M" = "100M"
    )
  ) +
  
labs(
  x = "Training data size (models) / developmental stage (humans)",
  
  y = paste0(
    "Accuracy (%)<br>",
    "<span style='font-size:9pt;'>",
    "Transformer/LSTM: PoSH sub-score; ",
    "Human: psycholinguistic experiments",
    "</span>"
  ),
  
  colour = NULL,
  fill = NULL,
  shape = NULL
) +
theme_classic(
  base_size = 12
) +
  
  theme(
    plot.margin = margin(
      8, 12, 15, 8
    ),
    
    axis.title.y = element_markdown(),
    
    panel.grid.major.y =
      element_line(
        colour = "grey88",
        linewidth = 0.35
      ),
    
    panel.border =
      element_rect(
        colour = "black",
        fill = NA,
        linewidth = 0.6
      ),
    
    strip.background =
      element_rect(
        fill = "grey92",
        colour = "black"
      ),
    
    strip.text =
      element_text(
        face = "bold",
        size = 11
      ),
    
    axis.text.x =
      element_text(
        face = "bold",
        colour = "#023FA5"
      ),
    
    legend.position = "bottom",
    
    legend.box.spacing = unit(
      0,
      "pt"
    ),
    
    panel.spacing = unit(
      1,
      "lines"
    )
  )


print(p)


ggsave(
  "human-model-compare.pdf",
  p,
  width = 6.83,
  height = 5.03
)