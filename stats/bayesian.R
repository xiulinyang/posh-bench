options(brms.backend="cmdstanr")
suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
  library(tidyr)
})


raw_df <- read.csv("benchmark_item_level_results.csv")

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

contrasts(df$category) <- contr.sum(nlevels(df$category))
contrasts(df$filter)   <- contr.sum(2) # yes=1, no=-1
contrasts(df$source)   <- contr.sum(2) # baby=1, wiki=-1


family_used <- bernoulli(link = "logit")
# 
# fml <- bf(
#   val ~ (size_z + category) * filter + source +
#     (1 | dataset) + (1 | dataset:random_seed) +
#     (1 | phenomenon)
# )

fml <- bf(
  val ~ (size_z + category) * filter + source +
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
  file = "fit_bayesian",
  file_refit = "on_change",
  control = list(adapt_delta = 0.99, max_treedepth = 15)
)

print(summary(m_brm))

pp_check(m_brm, ndraws = 200)
