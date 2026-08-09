suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(posterior)
  library(bayesplot)
})

abbr_cat <- c(
  "Binding"            = "Binding",
  "Island"             = "Islands",
  "Question Formation" = "QF",
  "Wanna"              = "Wanna"
)

cat_map <- tibble(
  coef_id = c("category1", "category2", "category3", "category4"),
  category = c(
    "Binding",
    "Island",
    "Question Formation",
    "Wanna"
  ),
  category_pretty = c(
    "Binding",
    "Islands",
    "QF",
    "Wanna"
  )
)


raw_df <- read.csv("posh_item_level_results.csv")

df <- raw_df %>%
  mutate(
    val = ifelse(correct == "correct", 1, 0),
    size_num = as.numeric(data_size),
    size_z   = as.numeric(scale(size_num)),
    filter = factor(filter, levels = c("yes", "no")), 
    random_seed = factor(random_seed),
    source   = factor(data_source, levels = c("baby", "wiki")), 
    category = factor(category),
    phenomenon = factor(phenomenon),
    dataset = factor(benchmark) 
  ) %>%
  select(val, size_z, category, filter, source, dataset, random_seed, phenomenon)


bayessian2000 <-readRDS("fit_bayesian.rds")
print(summary(bayessian2000))

draws <- as_draws_df(bayessian2000) 
dim(draws)


print(levels(df$category))
print(levels(df$filter))
print(levels(df$source))

np <- nuts_params(bayessian2000)
div_n <- sum(np$Parameter == "divergent__" & np$Value == 1)
div_by_chain <- with(subset(np, Parameter == "divergent__"), tapply(Value, Chain, sum))
cat("Divergences (total):", div_n, "\n")
print(div_by_chain)



cat_levels <- levels(df$category)
stopifnot(length(cat_levels) == 4)

cat_levels_pretty <- ifelse(cat_levels %in% names(abbr_cat), abbr_cat[cat_levels], cat_levels)
names(cat_levels_pretty) <- cat_levels

# cat_map <- tibble(
#   coef_id = paste0("category", 1:4),
#   category = cat_levels,
#   category_pretty = unname(cat_levels_pretty[cat_levels])
# )

draws <- as_draws_df(bayessian2000)

# fixed-effect category columns (usually b_category1..b_category3)
cat_coef_cols <- grep("^b_category\\d+$", names(draws), value = TRUE)
# category × filter columns (usually b_category1:filter1..b_category3:filter1)
int_coef_cols <- grep("^b_category\\d+:filter1$", names(draws), value = TRUE)

# Expect 3 explicit + 1 implicit = 4 levels
stopifnot(length(cat_coef_cols) == 3)
stopifnot(length(int_coef_cols) == 3)

draws2 <- draws %>%
  mutate(
    b_category4 = -rowSums(across(all_of(cat_coef_cols))),
    `b_category4:filter1` = -rowSums(across(all_of(int_coef_cols)))
  )


summ_draws <- function(x) {
  tibble(
    estimate = median(x),
    conf.low = quantile(x, 0.025),
    conf.high = quantile(x, 0.975)
  )
}


df_cat_main <- draws2 %>%
  select(matches("^b_category\\d+$")) %>%
  pivot_longer(everything(), names_to = "coef", values_to = "value") %>%
  mutate(coef_id = sub("^b_(category\\d+)$", "\\1", coef)) %>%
  left_join(cat_map, by = "coef_id") %>%
  group_by(category_pretty) %>%
  summarise(
    estimate = median(value),
    conf.low = quantile(value, 0.025),
    conf.high = quantile(value, 0.975),
    .groups = "drop"
  ) %>%
  mutate(
    group = "Main effect",
    term = paste0("Category (", category_pretty, ")")
  ) %>%
  select(term, estimate, conf.low, conf.high, group)

# 3.2 Filter main effect (sum coding: filter1 is half-difference yes vs no)
df_filt <- draws2 %>%
  transmute(value = .data$b_filter1) %>%
  summarise(
    estimate = median(value),
    conf.low = quantile(value, 0.025),
    conf.high = quantile(value, 0.975)
  ) %>%
  mutate(group = "Main effect", term = "Filtering (yes vs. no)") %>%
  select(term, estimate, conf.low, conf.high, group)

# 3.3 Source main effect (sum coding)
df_src <- draws2 %>%
  transmute(value = .data$b_source1) %>%
  summarise(
    estimate = median(value),
    conf.low = quantile(value, 0.025),
    conf.high = quantile(value, 0.975)
  ) %>%
  mutate(group = "Main effect", term = "Data (baby vs. wiki)") %>%
  select(term, estimate, conf.low, conf.high, group)

# 3.4 Size_z main effect
df_size <- draws2 %>%
  transmute(value = .data$b_size_z) %>%
  summarise(
    estimate = median(value),
    conf.low = quantile(value, 0.025),
    conf.high = quantile(value, 0.975)
  ) %>%
  mutate(group = "Main effect", term = "Data size (z scored)") %>%
  select(term, estimate, conf.low, conf.high, group)

# 3.5 Size × filter interaction
df_int_size <- draws2 %>%
  transmute(value = .data$`b_size_z:filter1`) %>%
  summarise(
    estimate = median(value),
    conf.low = quantile(value, 0.025),
    conf.high = quantile(value, 0.975)
  ) %>%
  mutate(group = "Interaction", term = "Filtering in data sizes") %>%
  select(term, estimate, conf.low, conf.high, group)

# 3.6 Category × filter interaction INCLUDING hidden category4
df_cat_filt <- draws2 %>%
  select(matches("^b_category\\d+:filter1$")) %>%
  pivot_longer(everything(), names_to = "coef", values_to = "value") %>%
  mutate(coef_id = sub("^b_(category\\d+):filter1$", "\\1", coef)) %>%
  left_join(cat_map, by = "coef_id") %>%
  group_by(category_pretty) %>%
  summarise(
    estimate = median(value),
    conf.low = quantile(value, 0.025),
    conf.high = quantile(value, 0.975),
    .groups = "drop"
  ) %>%
  mutate(
    group = "Interaction",
    term = paste0("Filtering in ", category_pretty)
  ) %>%
  select(term, estimate, conf.low, conf.high, group)

# =========================
# 4) Combine and plot
# =========================
plot_data <- bind_rows(
  df_cat_filt,  # put interactions first (top facet)
  df_int_size,
  df_cat_main,
  df_filt,
  df_src,
  df_size
) %>%
  mutate(
    group = factor(group, levels = c("Interaction", "Main effect"))
  ) %>%
  group_by(group) %>%
  arrange(estimate, .by_group = TRUE) %>%
  ungroup() %>%
  mutate(term = factor(term, levels = rev(unique(term))))  # top-to-bottom ordering

p_forest <- ggplot(plot_data, aes(x = estimate, y = term, shape = group)) +
  geom_segment(aes(x = conf.low, xend = conf.high, yend = term), linewidth = 0.6) +
  geom_point(size = 2.4) +
  geom_vline(xintercept = 0, linetype = 2, linewidth = 0.4) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y") +
  scale_shape_manual(values = c("Main effect" = 16, "Interaction" = 1)) +
  labs(
    x = "Posterior median (95% credible interval)",
    y = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(
    strip.text.y = element_text(face = "bold", size = 12, angle = 270),
    strip.background = element_rect(fill = "grey90", color = "grey20"),
    axis.text.y = element_text(hjust = 1, size = 12, face = "bold", margin = margin(r = 4)),
    axis.title.x = element_text(size = 12),
    panel.grid.major.y = element_line(color = "grey92"),
    panel.grid.minor = element_blank(),
    panel.spacing.y = unit(8, "pt"),
    legend.position = "none"
  )

print(p_forest)

# ggsave(
#   filename = "bayesian_transformer.pdf",
#   plot = p_forest,
#   device = "pdf",
#   width = 8.10,
#   height = 3.86,
#   units = "in"
# )

cat4_int <- df_cat_filt %>% filter(str_detect(term, paste0("in ", cat_map$category_pretty[4], "$")))
print(cat4_int)

df_cat_main %>%
  arrange(match(term,
                c("Category (Binding)",
                  "Category (Islands)",
                  "Category (QF)",
                  "Category (Wanna)"))) %>%
  print(n = Inf)


df_cat_filt %>%
  arrange(match(term,
                c("Filtering in Binding",
                  "Filtering in Islands",
                  "Filtering in QF",
                  "Filtering in Wanna"))) %>%
  print(n = Inf)


library(dplyr)
library(knitr)
library(kableExtra)

# -------------------------
# Group-level SDs
# -------------------------

s <- summary(bayessian2000)

df_sd <- tibble(
  Predictor = c(
    "Phenomenon intercept SD",
    "Random-seed intercept SD"
  ),
  Estimate = c(
    s$random$phenomenon$Estimate,
    s$random$random_seed$Estimate
  ),
  Lower = c(
    s$random$phenomenon$`l-95% CI`,
    s$random$random_seed$`l-95% CI`
  ),
  Upper = c(
    s$random$phenomenon$`u-95% CI`,
    s$random$random_seed$`u-95% CI`
  )
)

main_tab <- bind_rows(
  
  tibble(
    Predictor="Intercept",
    Estimate=summary(bayessian2000)$fixed["Intercept","Estimate"],
    Lower=summary(bayessian2000)$fixed["Intercept","l-95% CI"],
    Upper=summary(bayessian2000)$fixed["Intercept","u-95% CI"]
  ),
  
  tibble(
    Predictor="Training size (z)",
    Estimate=df_size$estimate,
    Lower=df_size$conf.low,
    Upper=df_size$conf.high
  ),
  
  df_cat_main %>%
    transmute(
      Predictor=term,
      Estimate=estimate,
      Lower=conf.low,
      Upper=conf.high
    ),
  
  tibble(
    Predictor="Filtering (yes vs. no)",
    Estimate=df_filt$estimate,
    Lower=df_filt$conf.low,
    Upper=df_filt$conf.high
  ),
  
  tibble(
    Predictor="Training source (baby vs. wiki)",
    Estimate=df_src$estimate,
    Lower=df_src$conf.low,
    Upper=df_src$conf.high
  )
)

int_tab <- bind_rows(
  
  tibble(
    Predictor="Training size (z) × Filtering",
    Estimate=df_int_size$estimate,
    Lower=df_int_size$conf.low,
    Upper=df_int_size$conf.high
  ),
  
  df_cat_filt %>%
    mutate(
      Predictor=gsub(
        "Filtering in ",
        "",
        term
      )
    ) %>%
    mutate(
      Predictor=paste0(Predictor," × Filtering")
    ) %>%
    select(Predictor,Estimate=estimate,Lower=conf.low,Upper=conf.high)
  
)


fmt <- function(x) sprintf("%.2f", x)

main_tab <- main_tab %>%
  mutate(
    Estimate=fmt(Estimate),
    `95\\% CrI`=paste0(
      "[",
      fmt(Lower),
      ", ",
      fmt(Upper),
      "]"
    )
  ) %>%
  select(Predictor,Estimate,`95\\% CrI`)

int_tab <- int_tab %>%
  mutate(
    Estimate=fmt(Estimate),
    `95\\% CrI`=paste0(
      "[",
      fmt(Lower),
      ", ",
      fmt(Upper),
      "]"
    )
  ) %>%
  select(Predictor,Estimate,`95\\% CrI`)

df_sd <- df_sd %>%
  mutate(
    Estimate=fmt(Estimate),
    `95\\% CrI`=paste0(
      "[",
      fmt(Lower),
      ", ",
      fmt(Upper),
      "]"
    )
  ) %>%
  select(Predictor,Estimate,`95\\% CrI`)


kbl(
  bind_rows(
    main_tab,
    tibble(
      Predictor="\\midrule\n\\textit{Interaction Effects}",
      Estimate="",
      `95\\% CrI`=""
    ),
    int_tab,
    tibble(
      Predictor="\\midrule\n\\textit{Group-level standard deviations}",
      Estimate="",
      `95\\% CrI`=""
    ),
    df_sd
  ),
  format="latex",
  booktabs=TRUE,
  escape=FALSE,
  align="lcc"
)
