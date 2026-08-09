suppressPackageStartupMessages({
  library(tidyverse)
  library(stringr)
  library(patchwork)
  library(grid)
})

BASE_DIR <- "working-mem"

RUN_BASES <- c(
  "results_TPT_10Mf_"      = "TPT",
  "results_dynamic_dyck_chunk_"  = "dynamic-dyck",
  "results_dynamic_only_chunk_"  = "dynamic-only",
  "results_linear_dyck_chunk_"   = "linear-dyck",
  "results_linear_only_chunk_"   = "linear-only"
)

DATASETS_ALL <- c(
  "BLiMP"               = "blimp_results",
  "Zorro"               = "zorro_results",
  "SCaMP (plausible)"   = "scamp_plausible_results",
  "SCaMP (implausible)" = "scamp_implausible_results",
  "PoSH"                = "posh_results"
)

DATASETS_POSH <- c("PoSH" = "posh_results")

CATEGORY_ORDER <- c("Island", "Question Formation", "Binding", "Wanna")
CHANCE_Y <- 0.5
MAX_EPOCH <- 10


COLOR_MAP <- c(
  "PoSH_nat"                = "#00897B",
  "PoSH_pre"                = "#80CBC4",
  "BLiMP_nat"               = "#E67E22",
  "BLiMP_pre"               = "#F7DC6F",
  "Zorro_nat"               = "#C2185B",
  "Zorro_pre"               = "#F48FB1",
  "SCaMP (plausible)_nat"   = "#2E7D32",
  "SCaMP (plausible)_pre"   = "#A5D6A7",
  "SCaMP (implausible)_nat" = "#7B1FA2",
  "SCaMP (implausible)_pre" = "#CE93D8"
)

BENCHMARK_KEYS <- c("PoSH", "BLiMP", "Zorro", "SCaMP (plausible)", "SCaMP (implausible)")

ATTENTION_LEVELS <- c("dynamic recency", "linear", "TPT")
TRAINING_LEVELS <- c("Dyck pretraining", "No pretraining")


LINETYPE_MAP <- c(
  "linear" = "solid",
  "dynamic recency" = "dashed",
  "TPT" = "dotdash"
)


category_dataset_map <- list(
  "Island" = list(
    blimp = c("adjunct_island", "wh_island", "complex_NP_island"),
    scamp = c("complex_np_island", "wh_island", "adjunct_island"),
    zorro = c("island-effects-adjunct_island"),
    posh  = c("island-adjunct", "island-complex-np", "island-wh")
  ),
  "Question Formation" = list(
    posh = c("question-formation_or", "question-formation_rr", "question-formation_sr")
  ),
  "Wanna" = list(posh = c("wanna")),
  "Binding" = list(
    blimp = c("principle_A_c_command","principle_A_case_2", "principle_A_domain_2", "principle_A_domain_3"),
    zorro = c("binding-principle_a"),
    scamp = c("principle_A_domain_1","principle_A_domain_3", "principle_A_c_command"),
    posh  = c("principle_a_command", "principle_a_locality")
  )
)

normalize_phen <- function(x) {
  x <- str_trim(as.character(x))
  x <- str_replace(x, "\\.(jsonl?|txt|csv|tsv)$", "")
  x <- str_replace_all(x, "-", "_")
  x
}

cat_lookup <- tibble(dataset_key = character(), phenomenon = character(), category = character())
for (cat in names(category_dataset_map)) {
  dm <- category_dataset_map[[cat]]
  for (dsk in names(dm)) {
    cat_lookup <- bind_rows(
      cat_lookup,
      tibble(dataset_key = dsk, phenomenon = normalize_phen(dm[[dsk]]), category = cat)
    )
  }
}

ds_key_for_map <- function(ds_pretty) {
  recode(ds_pretty,
         "BLiMP"               = "blimp",
         "Zorro"               = "zorro",
         "SCaMP (plausible)"   = "scamp",
         "SCaMP (implausible)" = "scamp",
         "PoSH"                = "posh",
         .default = "unknown")
}


parse_epoch_seed <- function(stem) {
  parts <- str_split(stem, "_", simplify = TRUE) |> as.character()
  epoch <- NA_integer_
  seed  <- NA_integer_

  idx_epoch <- which(parts == "epoch")
  idx_ckpt  <- which(parts == "epoch")
  idx <- if (length(idx_epoch) > 0) idx_epoch[1] else if (length(idx_ckpt) > 0) idx_ckpt[1] else NA_integer_

  if (!is.na(idx)) {
    if (idx + 1 <= length(parts) && str_detect(parts[idx + 1], "^\\d+$")) epoch <- as.integer(parts[idx + 1])
    if (idx - 1 >= 1 && str_detect(parts[idx - 1], "^\\d+$")) {
      seed <- as.integer(parts[idx - 1])
    } else if (idx + 2 <= length(parts) && str_detect(parts[idx + 2], "^\\d+$")) {
      seed <- as.integer(parts[idx + 2])
    }
  } else {
    m <- str_match(stem, "_epoch_?(\\d+)(?:_(\\d+))?$")
    if (!is.na(m[1,2])) epoch <- as.integer(m[1,2])
    if (!is.na(m[1,3])) seed  <- as.integer(m[1,3])
  }
  list(epoch = epoch, seed = seed)
}

read_epoch_file <- function(path) {
  df <- suppressWarnings(readr::read_csv(path, col_names = FALSE, show_col_types = FALSE))
  if (ncol(df) == 1) {
    parts <- str_split_fixed(as.character(df[[1]]), ",", 2)
    df <- tibble(X1 = parts[,1], X2 = parts[,2])
  } else {
    df <- df[,1:2]
    names(df) <- c("X1","X2")
  }

  df %>%
    mutate(phenomenon = as.character(X1), score = as.character(X2)) %>%
    filter(!str_detect(str_to_lower(str_trim(score)), "^best$")) %>%
    filter(!str_detect(str_to_lower(str_trim(phenomenon)), "^phenomenon$")) %>%
    filter(!str_detect(str_to_lower(str_trim(phenomenon)), "^unnamed:\\s*0$")) %>%
    filter(!str_detect(str_to_lower(phenomenon), "strict|pref")) %>%
    filter(!str_detect(str_to_lower(phenomenon), "semantics|pref")) %>%
    mutate(
      score = suppressWarnings(as.numeric(score)),
      phenomenon = normalize_phen(phenomenon)
    ) %>%
    drop_na(score) %>%
    select(phenomenon, score)
}

find_epoch_files <- function(ds_dir, run_prefix) {
  exts <- c("csv","tsv","txt")
  files <- map(exts, ~ list.files(
    ds_dir,
    pattern = paste0("^", run_prefix, ".*\\.", .x, "$"),
    full.names = TRUE,
    recursive = FALSE
  )) %>% unlist()


  if (length(files) == 0) return(tibble(epoch = integer(), seed = integer(), path = character()))

  tibble(path = files) %>%
    mutate(
      stem = tools::file_path_sans_ext(basename(path)),
      parsed = map(stem, parse_epoch_seed),
      epoch = map_int(parsed, "epoch"),
      seed  = map_int(parsed, "seed")
    ) %>%
    filter(!is.na(epoch)) %>%
    distinct(epoch, seed, path, .keep_all = TRUE) %>%
    arrange(epoch, if_else(is.na(seed), -1L, seed))
}

tcrit_from_count <- function(n) {
  df <- pmax(n - 1, 1)
  qt(0.975, df = df)
}

summarise_over_seeds <- function(df, value_col = "mean") {
  df %>%
    group_by(epoch) %>%
    summarise(
      mean_over_seeds  = mean(.data[[value_col]]),
      std_over_seeds   = sd(.data[[value_col]]),
      count_over_seeds = sum(!is.na(.data[[value_col]])),
      .groups = "drop"
    ) %>%
    mutate(
      std_over_seeds = replace_na(std_over_seeds, 0),
      sem = std_over_seeds / sqrt(pmax(count_over_seeds, 1)),
      tcrit = tcrit_from_count(count_over_seeds),
      ci_half = tcrit * sem
    )
}


build_per_category_df_posh_only <- function() {
  rows <- list()
  ds_pretty <- "PoSH"
  ds_dir <- file.path(BASE_DIR, DATASETS_POSH[[ds_pretty]])
  if (!dir.exists(ds_dir)) stop("posh_results dir not found.")
  dsk <- ds_key_for_map(ds_pretty)

  for (run_prefix in names(RUN_BASES)) {
    run_key <- RUN_BASES[[run_prefix]]
    ep_tbl <- find_epoch_files(ds_dir, run_prefix)
    if (nrow(ep_tbl) == 0) next

    for (i in seq_len(nrow(ep_tbl))) {
      epoch <- ep_tbl$epoch[i]; seed <- ep_tbl$seed[i]; fpath <- ep_tbl$path[i]
      df0 <- read_epoch_file(fpath)
      if (nrow(df0) == 0) next

      df2 <- df0 %>%
        mutate(dataset_key = dsk) %>%
        left_join(cat_lookup, by = c("dataset_key", "phenomenon")) %>%
        drop_na(category) %>%
        filter(category %in% CATEGORY_ORDER)

      if (nrow(df2) == 0) next

      df_cat <- df2 %>%
        group_by(category) %>%
        summarise(mean = mean(score), .groups = "drop") %>%
        mutate(dataset = "PoSH", run_key = run_key, epoch = epoch, seed = seed)

      rows[[length(rows) + 1]] <- df_cat
    }
  }

  if (length(rows) == 0) stop("No PoSH data.")
  bind_rows(rows)
}

build_overall_df_all <- function() {
  rows <- list()
  for (ds_pretty in names(DATASETS_ALL)) {
    ds_dir <- file.path(BASE_DIR, DATASETS_ALL[[ds_pretty]])
    if (!dir.exists(ds_dir)) next

    for (run_prefix in names(RUN_BASES)) {
      run_key <- RUN_BASES[[run_prefix]]
      ep_tbl <- find_epoch_files(ds_dir, run_prefix)
      if (nrow(ep_tbl) == 0) next

      for (i in seq_len(nrow(ep_tbl))) {
        epoch <- ep_tbl$epoch[i]; seed <- ep_tbl$seed[i]; fpath <- ep_tbl$path[i]
        df0 <- read_epoch_file(fpath)
        if (nrow(df0) == 0) next

        rows[[length(rows) + 1]] <- tibble(
          dataset = ds_pretty, run_key = run_key, epoch = epoch, seed = seed, mean = mean(df0$score)
        )
      }
    }
  }

  if (length(rows) == 0) stop("No overall data.")
  bind_rows(rows)
}


plot_row1_posh_row2_all <- function(df_cat_posh, df_overall_all) {

  df_cat_posh    <- df_cat_posh %>% filter(epoch <= MAX_EPOCH)
  df_overall_all <- df_overall_all %>% filter(epoch <= MAX_EPOCH)

  process_df <- function(df) {
    df %>%
      mutate(
        rk = as.character(run_key),
        attention = case_when(
          rk == "TPT"                ~ "TPT",
          str_detect(rk, "^dynamic") ~ "dynamic recency",
          TRUE                       ~ "linear"
        ),
        attention = factor(attention, levels = ATTENTION_LEVELS),
        training = if_else(str_detect(rk, "dyck"), "Dyck pretraining", "No pretraining"),
        training = factor(training, levels = TRAINING_LEVELS),
        color_group = if_else(training == "Dyck pretraining", paste0(dataset, "_pre"), paste0(dataset, "_nat"))
      )
  }

  smooth_curve <- function(df, n = 80) {
    xs <- df$epoch; ys <- df$mean_over_seeds
    if (length(unique(xs)) < 4) return(df)
    x_new <- seq(min(xs), max(xs), length.out = n)
    y_new <- spline(xs, ys, xout = x_new, method = "natural")$y
    out <- tibble(epoch = x_new, mean_over_seeds = y_new)
    carry <- setdiff(names(df), c("epoch", "mean_over_seeds"))
    for (cc in carry) out[[cc]] <- df[[cc]][1]
    out
  }

  my_theme <- theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      strip.background = element_rect(fill = "gray93", color = NA),
      strip.text = element_text(face = "bold", size = 9),
      axis.text = element_text(color = "black", size = 8),
      axis.title = element_text(color = "black", size = 10)
    )

  # ----- Row 1 data -----
  df_cat_sum <- df_cat_posh %>%
    group_by(category, run_key) %>%
    group_modify(~ summarise_over_seeds(.x, "mean")) %>%
    ungroup() %>%
    mutate(dataset = "PoSH") %>%
    process_df() %>%
    mutate(
      category = factor(category, levels = CATEGORY_ORDER),
      lower = mean_over_seeds - std_over_seeds,
      upper = mean_over_seeds + std_over_seeds
    )

  df_cat_line <- df_cat_sum %>%
    group_by(category, run_key, attention, training, color_group) %>%
    group_modify(~ smooth_curve(.x)) %>%
    ungroup()


  one_point_top <- df_cat_line %>%
    group_by(category, run_key, attention, training, color_group) %>%
    summarise(n_points = n(), epochs = paste(sort(unique(epoch)), collapse = ","), .groups = "drop") %>%
    filter(n_points == 1) %>%
    arrange(category, run_key, attention, training)

  message("---- one-point groups (TOP row) ----")
  print(one_point_top, n = 200)


  df_cat_line2 <- df_cat_line %>%
    group_by(category, run_key, attention, training, color_group) %>%
    filter(n() >= 2) %>%
    ungroup()

  make_one_cat_plot <- function(cat_name, show_y = FALSE) {
    df_s <- df_cat_sum  %>% filter(category == cat_name) %>% mutate(facet = as.character(cat_name))
    df_l <- df_cat_line2 %>% filter(category == cat_name) %>% mutate(facet = as.character(cat_name))

    ggplot(df_s, aes(epoch, mean_over_seeds)) +
      geom_ribbon(aes(ymin = lower, ymax = upper, group = run_key, fill = color_group),
                  alpha = 0.1, color = NA) +
      geom_line(data = df_l,
                aes(group = run_key, linetype = attention, color = color_group),
                linewidth = 0.8) +
      geom_hline(yintercept = CHANCE_Y, linetype = "dashed", color = "gray60", linewidth = 0.4) +
      facet_wrap(~ facet, nrow = 1) +
      scale_color_manual(values = COLOR_MAP, guide = "none") +
      scale_fill_manual(values = COLOR_MAP, guide = "none") +
      scale_linetype_manual(values = LINETYPE_MAP, guide = "none") +
      labs(y = if (show_y) "Accuracy" else NULL) +
      my_theme +
      theme(
        panel.background = element_rect(fill = "white", color = NA),
        axis.title.x = element_blank(),
        axis.text.x  = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.y = if (show_y) element_text(size = 10) else element_blank(),
        axis.text.y  = element_text(color = "black", size = 8),
        axis.ticks.y = element_line()
      )
  }

  p_island <- make_one_cat_plot("Island", show_y = TRUE)
  p_qf     <- make_one_cat_plot("Question Formation", show_y = FALSE)
  p_bind   <- make_one_cat_plot("Binding", show_y = FALSE)
  p_wanna  <- make_one_cat_plot("Wanna", show_y = FALSE)

  # ----- Row 2 data -----
  df_over_sum <- df_overall_all %>%
    group_by(dataset, run_key) %>%
    group_modify(~ summarise_over_seeds(.x, "mean")) %>%
    ungroup() %>%
    process_df() %>%
    mutate(
      dataset = factor(dataset, levels = BENCHMARK_KEYS),
      lower = mean_over_seeds - std_over_seeds,
      upper = mean_over_seeds + std_over_seeds
    )

  df_over_line <- df_over_sum %>%
    group_by(dataset, run_key, attention, training, color_group) %>%
    group_modify(~ smooth_curve(.x)) %>%
    ungroup()


  one_point_bottom <- df_over_line %>%
    group_by(dataset, run_key, attention, training, color_group) %>%
    summarise(n_points = n(), epochs = paste(sort(unique(epoch)), collapse = ","), .groups = "drop") %>%
    filter(n_points == 1) %>%
    arrange(dataset, run_key, attention, training)

  message("---- one-point groups (BOTTOM row) ----")
  print(one_point_bottom, n = 200)

  df_over_line2 <- df_over_line %>%
    group_by(dataset, run_key, attention, training, color_group) %>%
    filter(n() >= 2) %>%
    ungroup()

  p_bot <- ggplot(df_over_sum, aes(epoch, mean_over_seeds)) +
    geom_ribbon(aes(ymin = lower, ymax = upper, group = interaction(dataset, run_key), fill = color_group),
                alpha = 0.1, color = NA) +
    geom_line(data = df_over_line2,
              aes(group = interaction(dataset, run_key), color = color_group, linetype = attention),
              linewidth = 0.8) +
    geom_hline(yintercept = CHANCE_Y, linetype = "dashed", color = "gray60", linewidth = 0.4) +
    facet_wrap(~ dataset, ncol = 5, scales = "free_y") +
    scale_color_manual(values = COLOR_MAP, guide = "none") +
    scale_fill_manual(values = COLOR_MAP, guide = "none") +
    scale_linetype_manual(values = LINETYPE_MAP, guide = "none") +
    labs(x = "Epoch", y = "Accuracy") +
    my_theme

  # ----- Legends -----
  get_leg <- function(p) {
    g <- ggplotGrob(p)
    g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
  }

  leg_theme_right <- theme_void() +
    theme(
      legend.position = "right",
      legend.direction = "vertical",
      legend.box = "vertical",
      legend.justification = "left",
      legend.box.just = "left",
      legend.title = element_text(size = 13, face = "bold", hjust = 0),
      legend.text  = element_text(size = 13, hjust = 0),
      legend.key.width  = unit(1.0, "lines"),
      legend.key.height = unit(0.6, "lines"),
      plot.margin = margin(0, 0, 0, 0)
    )

  p_att <- ggplot() +
    geom_line(
      data = tibble(a = factor(ATTENTION_LEVELS, levels = ATTENTION_LEVELS), x = 1:3, y = 1),
      aes(x, y, linetype = a),
      color = "black",
      linewidth = 0.8
    ) +
    scale_linetype_manual(values = LINETYPE_MAP, name = "Attention") +
    guides(linetype = guide_legend(ncol = 1)) +
    leg_theme_right

  p_train <- ggplot() +
    geom_line(
      data = tibble(t = factor(TRAINING_LEVELS, levels = TRAINING_LEVELS), x = 1:2, y = 1),
      aes(x, y, color = t),
      linewidth = 3
    ) +
    scale_color_manual(
      values = c("Dyck pretraining" = "gray70", "No pretraining" = "gray20"),
      name = "Training"
    ) +
    guides(color = guide_legend(ncol = 1)) +
    leg_theme_right

  legend_top <- (wrap_elements(get_leg(p_att)) / wrap_elements(get_leg(p_train))) +
    plot_layout(heights = c(1, 1))

  LEGEND_DOTS <- c(
    "PoSH" = COLOR_MAP[["PoSH_nat"]],
    "BLiMP" = COLOR_MAP[["BLiMP_nat"]],
    "Zorro" = COLOR_MAP[["Zorro_nat"]],
    "SCaMP (plausible)" = COLOR_MAP[["SCaMP (plausible)_nat"]],
    "SCaMP (implausible)" = COLOR_MAP[["SCaMP (implausible)_nat"]]
  )

  leg_theme_bench <- theme_void() +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.justification = "center",
      legend.box.just = "center",
      legend.title = element_text(size = 13, face = "bold"),
      legend.text  = element_text(size = 13),
      legend.key.width  = unit(1.2, "lines"),
      legend.key.height = unit(0.6, "lines"),
      plot.margin = margin(0, 0, 0, 0)
    )

  p_b <- ggplot(
    tibble(d = factor(names(LEGEND_DOTS), levels = names(LEGEND_DOTS)), x = 1, y = 1),
    aes(x, y, color = d)
  ) +
    geom_point(size = 3) +
    scale_color_manual(values = LEGEND_DOTS, name = "Benchmarks") +
    guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
    leg_theme_bench

  legend_bottom <- wrap_elements(get_leg(p_b))

  # Layout
  row1 <- (p_island | p_qf | p_bind | p_wanna) | legend_top
  (row1 / p_bot / legend_bottom) +
    plot_layout(heights = c(1, 1, 0), widths = c(1, 1, 1, 1, 0.2))
}


df_cat_posh <- build_per_category_df_posh_only()
df_overall_all <- build_overall_df_all()

p <- plot_row1_posh_row2_all(df_cat_posh, df_overall_all)
print(p)

ggsave(file.path(BASE_DIR, "working_memory.pdf"), p, width = 10.25, height = 4.45)