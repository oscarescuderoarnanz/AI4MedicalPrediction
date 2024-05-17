
### Machine Learning for diagnosis and prognosis of blood-related infections

> Paste brief description here.

### Getting started

> Paste content.

### Background

> Paste content.

### Datasets

[url-eicu-article]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6132188/
[url-eicu-physionet]: https://physionet.org/content/eicu-crd/2.0/
[url-eicu-documentation]: https://eicu-crd.mit.edu/
[url-mimic-nature]: https://www.nature.com/articles/s41597-022-01899-x/
[url-mimic-physionet]: https://physionet.org/content/mimiciv/2.2/
[url-hirid-arxiv]: https://arxiv.org/abs/2111.08536/
[url-hirid-documentation]: https://hirid.intensivecare.ai/
[url-hirid-physionet]: https://physionet.org/content/hirid/1.1.1/
[url-hirid-github-benchmark]: https://github.com/ratschlab/HIRID-ICU-Benchmark
[url-aumcdb]: https://amsterdammedicaldatascience.nl/#amsterdamumcdb/
[url-aumcdb-documentation]: https://github.com/AmsterdamUMC/AmsterdamUMCdb/wiki
[url-aumcdb-sepsis3-github]: https://github.com/tedinburgh/sepsis3-amsterdamumcdb
[url-aumcdb-sepsis3-article]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9650242/


[url-ricu-pdf]: https://cran.r-project.org/web/packages/ricu/vignettes/ricu.pdf
[url-ricu-doc]: https://rdrr.io/cran/ricu/man/id_tbl.html
[url-ricu-doc1]: https://eth-mds.github.io/ricu/reference/index.html
[url-ricu-doc2]: https://rdrr.io/cran/ricu/src/R/setup-download.R
[url-ricu-doc3]: https://cran.r-project.org/web/packages/ricu/ricu.pdf

[url-moor2023]: https://pubmed.ncbi.nlm.nih.gov/37588623/
[url-moor2023-sm]: https://ars.els-cdn.com/content/image/1-s2.0-S2589537023003012-mmc1.pdf
[url-moor2023-github]: https://github.com/BorgwardtLab/multicenter-sepsis

| Name | Website | Docs | Other |
| ---    | --- | --- | --- | 
| MIMIC  | [Physionet][url-mimic-physionet] | -- | [Article (description)][url-mimic-nature] |
| eICU   | [Physionet][url-eicu-physionet]  | [Link][url-eicu-documentation] | [Article (description)][url-eicu-article] |
| HiRID  | [Physionet][url-hirid-physionet] | [Link][url-hirid-documentation] | [ICU Benchmark][url-hirid-github-benchmark] |
| AUMCdb | [Website][url-aumcdb] | -- | [Link][url-aumcdb-documentation] |

Excellent study on 'Predicting sepsis using deep learning across international sites: a retrospective development 
and validation study' from Michael Moor et al, published in eClinicalMedicine, 2023 ([Article][url-moor2023], 
[Supplementary Material][url-moor2023-sm], [Repository][url-moor2023-github])

#### MIMIMC
#### HiRID
#### eICU
#### AUMCdb

- Calculating the sepsis3 criteria in AUMCdb ([Article][url-aumcdb-sepsis3-article], [Repository][url-aumcdb-sepsis3-github])

### RICU

> The code as it is in the repository does not work. This are just some dingings.
>

for (x in c("mimic")) {

  if (!is_data_avail(x)) {
    msg("setting up `{x}`\n")
    setup_src_data(x)
  }

  dir <- file.path("./")
  msg("exporting data for `{x}`\n")
  export_data(x, dest_dir = dir)
}


for (x in c("miiv")) {
  if (!is_data_avail(x)) {
    msg("setting up `{x}`\n")
    setup_src_data(x)
  }
}

for (x in c("miiv")) {
    msg("setting up `{x}`\n")
    setup_src_data(x)
}


for (x in c("miiv")) {
  if (!is_data_avail(x)) {
    msg("setting up `{x}`\n")
    import_src(x)
  }
}

for (x in c("aumc")) {
  msg("exporting data for `{x}`\n")
  export_data(x)
}

dir <- file.path("data-export", paste0("eicu_bernard"))
export_data("eicu", dest_dir = dir)

install.packages(
 c("mimic.demo", "eicu.demo"),
 repos = "https://eth-mds.github.io/physionet-demo"
)
load_concepts("hr", demo, verbose = FALSE)
df <- load_concepts("hr", "eicu_demo", verbose = FALSE)
write_psv(df, './')


src='eicu'
dest_dir <- './'
atr <- list(
	ricu = list(
	  id_vars = id_vars(dat$dat), index_var = index_var(dat$dat),
	  time_unit = units(interval(dat$dat)), time_step = time_step(dat$dat)
	),
	mcsep = list(cohorts = dat$coh)
 )


  dat <- as.data.frame(dat)
  fil <- file.path(dest_dir, paste(src, packageVersion("ricu"), sep = "_"))

  jsonlite::write_json(atr$mcsep$splits,
    file.path(cfg_path("splits"), paste0(basename(fil), ".json")),
    pretty = TRUE
  )

  create_parquet(dat, fil, atr, chunk_size = 1e3)






Sys.setenv(
    RICU_PHYSIONET_USER = "bahp",
    RICU_PHYSIONET_PASS = "Imperial-5..",
    RICU_AUMC_TOKEN = "74ca2023-9aab-4f32-a3e6-ebec156b82ab"
)

### Contact

> Paste content.

### License

> Paste content.