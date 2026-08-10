options(brms.backend="cmdstanr")
suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
  library(tidyr)
})


raw_df <- read.csv("posh_item_level_results.csv")

df <- raw_df %>%
  mutate(
    val = ifelse(correct == "correct", 1, 0),
    size_num = as.numeric(data_size),
    size_z   = as.numeric(scale(size_num)),
    random_seed = factor(random_seed),
    category = factor(category),
    phenomenon = factor(phenomenon),
    dataset = factor(benchmark),
    source_filter = case_when(
      data_source == "wiki"              ~ "wiki",
      data_source == "baby" & filter == "yes" ~ "baby_filtered",
      data_source == "baby" & filter == "no"  ~ "baby_unfiltered",
      TRUE ~ NA_character_
    ),
    source_filter = factor(source_filter,
      levels = c("baby_unfiltered", "baby_filtered", "wiki"))
  ) %>%
  select(val, size_z, category, source_filter, dataset, random_seed, phenomenon)


contrasts(df$category) <- contr.sum(nlevels(df$category))
family_used <- bernoulli(link = "logit")

fml <- bf(
  val ~ (size_z + category) * source_filter +
    (1 | random_seed) + (1 | phenomenon)
)

priors <- c(
  set_prior("normal(0, 0.5)", class = "b"),
  set_prior("normal(0, 1.5)", class = "Intercept"),
  set_prior("exponential(2)", class = "sd")
)


m_brm <- brm(
  formula = fml,
  data = df,
  family = family_used,
  prior = priors,
  chains = 4,
   cores  = 4,
  threads = threading(1),
  backend="cmdstanr",
  refresh = 10,
  iter = 2000,
  warmup = 1000,
  seed = 42,
  file = "fit_bayesian_new",
  file_refit = "on_change",
  control = list(adapt_delta = 0.99, max_treedepth = 15)
)

print(summary(m_brm))

pp_check(m_brm, ndraws = 200)
