"""
=============================================================================
Serum proteomics pipeline: Responders (R) vs Non-responders (NR) to anti-PD-1
=============================================================================
Modelled on the analysis logic of Zila et al., Clin Cancer Res 2024
("Proteomic Profiling of Advanced Melanoma Patients to Predict Therapeutic
Response to Anti-PD-1 Therapy").

HOW TO USE
----------
The file is split into "# %%" cells. In VS Code / PyCharm / Spyder you can run
it cell by cell (like a Jupyter notebook). Each cell = one analysis step.

  STEP 0  Config + imports
  STEP 1  Load protein matrix + sample metadata  (or DEMO data)
  STEP 2  Clean the matrix (contaminants, decoys, duplicate genes)
  STEP 3  log2 transform + missing-value filter
  STEP 4  Normalisation (median centring)
  STEP 5  Imputation (down-shifted normal, Perseus style)
  STEP 6  Quality control (missingness, correlation, PCA)
  STEP 7  Differential abundance R vs NR (+ BH FDR)
  STEP 8  Volcano plot + heatmap
  STEP 9  Over-representation (pathway) analysis with a local GMT file
  STEP 10 Validation in an independent cohort (direction + significance)
  STEP 11 Literature overlap with published signatures
  STEP 12 Survival analysis in TCGA-SKCM (KM + log-rank + Cox)
  STEP 13 (optional extension) Predictive model with nested cross-validation

Requirements:
  pip install pandas numpy scipy matplotlib seaborn scikit-learn lifelines
  (lifelines only for STEP 12)

INPUT FORMATS EXPECTED
----------------------
protein matrix (CSV/TSV): rows = protein groups, columns = samples, values =
  raw (NOT log) intensities (LFQ / MaxLFQ / Spectronaut / DIA-NN PG.MaxLFQ).
  It must contain one identifier column, e.g. "Gene" (gene symbol).
  Zeros or NaN = not detected.
metadata (CSV): one row per sample with at least
  sample_id, response (R / NR), and optionally batch, cohort, age, sex, LDH...
  sample_id values must match the column names of the protein matrix.
=============================================================================
"""

# %% STEP 0 — Configuration and imports ------------------------------------
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")            # remove this line if you want pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ---- paths: edit these ----------------------------------------------------
DEMO = True                       # True -> synthetic data so you can test the pipeline
PROTEIN_FILE  = "discovery_proteins.tsv"
METADATA_FILE = "discovery_metadata.csv"
VALIDATION_PROTEIN_FILE  = "validation_proteins.tsv"   # optional (STEP 10)
VALIDATION_METADATA_FILE = "validation_metadata.csv"
GMT_FILE = "reactome_or_GO_BP.gmt"                     # optional (STEP 9)
TCGA_EXPR_FILE = "tcga_skcm_expression.tsv"            # optional (STEP 12)
TCGA_CLIN_FILE = "tcga_skcm_clinical.tsv"
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)

# ---- analysis parameters --------------------------------------------------
ID_COL = "Gene"          # identifier column in the protein matrix
GROUP_COL = "response"   # column in metadata
GROUP_A, GROUP_B = "NR", "R"   # log2FC is computed as A - B  (NR vs R)
MIN_VALID_FRAC = 0.7     # keep protein if >=70% valid values in >=1 group
IMPUTE_SHIFT, IMPUTE_WIDTH = 1.8, 0.3   # Perseus defaults
FDR_CUTOFF = 0.05
LOG2FC_CUTOFF = 0.5      # ~1.4-fold; serum effects are usually small
COVARIATES = []          # e.g. ["batch"] or ["batch", "age"]
RNG = np.random.default_rng(42)


# %% STEP 1 — Load data -------------------------------------------------------
def make_demo(n_prot=600, n_R=28, n_NR=28, seed=0, true_markers=None):
    """Synthetic serum-like data: log-normal intensities, MNAR missingness,
    and a handful of proteins truly higher in NR (acute-phase-like)."""
    rng = np.random.default_rng(seed)
    genes = [f"PROT{i}" for i in range(n_prot)]
    true_markers = true_markers or ["CRP", "SAA1", "SAA2", "S100A8", "S100A9",
                                    "LBP", "LYVE1", "CFHR3", "C1RL", "LDHB"]
    genes[:len(true_markers)] = true_markers
    samples = [f"S{i:03d}" for i in range(n_R + n_NR)]
    resp = ["R"] * n_R + ["NR"] * n_NR
    base = rng.normal(22, 3, n_prot)                     # protein abundance level
    X = base[:, None] + rng.normal(0, 0.8, (n_prot, len(samples)))
    nr = np.array(resp) == "NR"
    X[:len(true_markers)][:, nr] += rng.uniform(0.4, 1.0, (len(true_markers), 1))
    # missing not at random: low intensity -> more likely missing
    p_missing = 1 / (1 + np.exp((X - 19) * 1.5))
    X[rng.random(X.shape) < p_missing] = np.nan
    mat = pd.DataFrame(2 ** X, columns=samples)
    mat.insert(0, ID_COL, genes)
    # sprinkle contaminants for STEP 2
    mat.loc[n_prot - 3:, ID_COL] = ["KRT1", "KRT10", "TRYP_PIG"]
    meta = pd.DataFrame({"sample_id": samples, "response": resp,
                         "batch": rng.choice(["b1", "b2"], len(samples))})
    return mat, meta


def load_inputs(prot_file, meta_file):
    sep = "\t" if prot_file.endswith((".tsv", ".txt")) else ","
    mat = pd.read_csv(prot_file, sep=sep)
    meta = pd.read_csv(meta_file)
    return mat, meta


if DEMO:
    raw, meta = make_demo()
else:
    raw, meta = load_inputs(PROTEIN_FILE, METADATA_FILE)

meta = meta.set_index("sample_id")
samples = [s for s in meta.index if s in raw.columns]   # keep only matched samples
meta = meta.loc[samples]
print(f"Loaded {raw.shape[0]} protein groups x {len(samples)} samples")
print(meta[GROUP_COL].value_counts())


# %% STEP 2 — Clean the matrix ------------------------------------------------
# WHY: search engines report decoy hits (REV_), contaminants (CON_, keratins,
# trypsin) and protein groups without gene names. These must not enter stats.
def clean_matrix(df):
    df = df.copy()
    ids = df[ID_COL].astype(str)
    bad = (ids.str.startswith(("REV_", "CON_", "rev_", "con_"))
           | ids.str.match(r"^KRT\d")          # skin keratins = handling contamination
           | ids.str.contains("TRYP_")         # trypsin
           | ids.isin(["nan", ""]))
    # MaxQuant flags if present
    for flag in ["Reverse", "Potential contaminant", "Only identified by site"]:
        if flag in df.columns:
            bad |= df[flag].astype(str).eq("+")
    df = df.loc[~bad]
    # protein groups like "SAA1;SAA2" -> keep first gene as label
    df[ID_COL] = df[ID_COL].astype(str).str.split(";").str[0]
    # duplicated gene labels -> keep the row with the most valid values
    df["_n_valid"] = df[samples].replace(0, np.nan).notna().sum(axis=1)
    df = (df.sort_values("_n_valid", ascending=False)
            .drop_duplicates(ID_COL).drop(columns="_n_valid"))
    return df.set_index(ID_COL)[samples]

intens = clean_matrix(raw)
print(f"After cleaning: {intens.shape[0]} proteins")


# %% STEP 3 — log2 transform + missing-value filter -------------------------
# WHY log2: MS intensities are right-skewed and multiplicative; log2 makes them
# ~normal and makes a difference of 1 = 2-fold change.
# WHY filter: a protein seen in 3/56 samples can't be tested reliably. The usual
# rule is ">= X% valid values in at least ONE group" so that proteins present
# in NR but absent in R (biologically interesting!) are not thrown out.
log = np.log2(intens.replace(0, np.nan))

groups = meta[GROUP_COL]
def valid_frac(df, grp):
    return df.loc[:, groups[groups == grp].index].notna().mean(axis=1)

keep = (valid_frac(log, GROUP_A) >= MIN_VALID_FRAC) | (valid_frac(log, GROUP_B) >= MIN_VALID_FRAC)
log = log.loc[keep]
print(f"After valid-value filter: {log.shape[0]} proteins "
      f"({log.isna().mean().mean():.1%} missing remain)")


# %% STEP 4 — Normalisation ------------------------------------------------
# WHY: differences in loaded amount / depletion efficiency / MS sensitivity
# shift whole samples up or down. Median centring removes that, assuming most
# proteins do NOT change between groups (reasonable for serum).
def median_normalise(df):
    med = df.median(axis=0)
    return df - med + med.mean()

norm = median_normalise(log)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
log.boxplot(ax=ax[0], showfliers=False); ax[0].set_title("before normalisation")
norm.boxplot(ax=ax[1], showfliers=False); ax[1].set_title("after median normalisation")
for a in ax: a.set_xticklabels([]); a.set_ylabel("log2 intensity")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/01_normalisation.png", dpi=150); plt.close()


# %% STEP 5 — Imputation ----------------------------------------------------
# WHY: most stats need complete data. In proteomics, missing values are mostly
# MNAR (below detection limit), so we impute from a down-shifted, narrow normal
# distribution per sample (Perseus default: shift 1.8 SD, width 0.3 SD).
# ALWAYS also check that your top hits are not driven by imputed values
# (see 'pct_imputed' column in STEP 7).
def impute_downshift(df, shift=IMPUTE_SHIFT, width=IMPUTE_WIDTH, rng=RNG):
    out = df.copy()
    for c in out.columns:
        col = out[c]; mu, sd = col.mean(), col.std()
        n = col.isna().sum()
        out.loc[col.isna(), c] = rng.normal(mu - shift * sd, width * sd, n)
    return out

missing_mask = norm.isna()
imp = impute_downshift(norm)


# %% STEP 6 — Quality control ---------------------------------------------
# Look for: outlier samples, batch effects, whether R/NR separate at all.
# Small n (tens of patients) means PCA separation is often weak — that's normal.
color_map = {GROUP_A: "tab:red", GROUP_B: "tab:blue"}

# 6a) missingness per sample
plt.figure(figsize=(10, 3))
missing_mask.mean().plot.bar(color=[color_map[g] for g in groups])
plt.ylabel("fraction missing"); plt.xticks([])
plt.title("Missing values per sample (red=NR, blue=R)")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/02_missingness.png", dpi=150); plt.close()

# 6b) sample-sample Pearson correlation
corr = imp.corr()
sns.clustermap(corr, cmap="viridis", figsize=(8, 8),
               col_colors=groups.map(color_map), xticklabels=False, yticklabels=False)
plt.savefig(f"{OUTDIR}/03_sample_correlation.png", dpi=150); plt.close()
low = corr.mean().sort_values().head(3)
print("Least correlated samples (check for outliers):\n", low.round(3))

# 6c) PCA on z-scored proteins
z = imp.sub(imp.mean(axis=1), axis=0).div(imp.std(axis=1), axis=0)
pca = PCA(n_components=2).fit(z.T)
pcs = pd.DataFrame(pca.transform(z.T), index=imp.columns, columns=["PC1", "PC2"])
fig, ax = plt.subplots(1, 2 if "batch" in meta else 1, figsize=(11, 4.5), squeeze=False)
sns.scatterplot(data=pcs.join(meta), x="PC1", y="PC2", hue=GROUP_COL,
                palette=color_map, ax=ax[0, 0])
ax[0, 0].set_title("PCA coloured by response")
if "batch" in meta:
    sns.scatterplot(data=pcs.join(meta), x="PC1", y="PC2", hue="batch", ax=ax[0, 1])
    ax[0, 1].set_title("PCA coloured by batch  (should NOT dominate)")
for a in ax[0]:
    a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/04_pca.png", dpi=150); plt.close()


# %% STEP 7 — Differential abundance NR vs R -----------------------------------
# For each protein: log2FC, Welch t-test, Mann-Whitney U (robust check), and
# optionally a linear model with covariates (batch, age...). Then
# Benjamini-Hochberg FDR across all proteins (multiple testing!).
def bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked, 0, 1)
    return out

def lm_pvalue(y, design, coef_idx=1):
    """OLS per protein: y ~ intercept + group + covariates. Returns (beta, p)."""
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = design.shape[0] - design.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[coef_idx, coef_idx])
    t = beta[coef_idx] / se
    return beta[coef_idx], 2 * stats.t.sf(abs(t), dof)

def differential(data, meta, mask=None, covariates=COVARIATES):
    a = meta.index[meta[GROUP_COL] == GROUP_A]
    b = meta.index[meta[GROUP_COL] == GROUP_B]
    res = pd.DataFrame(index=data.index)
    res["log2FC"] = data[a].mean(axis=1) - data[b].mean(axis=1)
    res["p_welch"] = stats.ttest_ind(data[a], data[b], axis=1, equal_var=False).pvalue
    res["p_mwu"] = [stats.mannwhitneyu(data.loc[g, a], data.loc[g, b]).pvalue
                    for g in data.index]
    p_main = "p_welch"
    if covariates:
        X = pd.get_dummies(meta[covariates], drop_first=True).astype(float)
        design = np.column_stack([np.ones(len(meta)),
                                  (meta[GROUP_COL] == GROUP_A).astype(float), X.values])
        lm = [lm_pvalue(data.loc[g, meta.index].values, design) for g in data.index]
        res["log2FC_adj"], res["p_lm"] = zip(*lm)
        p_main = "p_lm"
    res["p"] = res[p_main]
    res["FDR"] = bh_fdr(res["p"])
    if mask is not None:   # how much of each protein was imputed?
        res["pct_imputed"] = mask.loc[data.index].mean(axis=1).round(2)
    res["significant"] = (res["FDR"] < FDR_CUTOFF) & (res["log2FC"].abs() >= LOG2FC_CUTOFF)
    res["direction"] = np.where(res["log2FC"] > 0, f"up_in_{GROUP_A}", f"up_in_{GROUP_B}")
    return res.sort_values("p")

de = differential(imp, meta, mask=missing_mask)
de.to_csv(f"{OUTDIR}/differential_abundance_discovery.csv")
hits = de[de["significant"]]
print(f"{len(hits)} significant proteins (FDR<{FDR_CUTOFF}, |log2FC|>={LOG2FC_CUTOFF})")
print(hits.head(15).round(4))
# Note: with ~56 patients, nominal p<0.05 lists (as many discovery papers use)
# are useful for hypothesis generation, but report FDR and rely on VALIDATION.


# %% STEP 8 — Volcano plot + heatmap of hits --------------------------------
plt.figure(figsize=(6.5, 5.5))
c = np.where(de["significant"], np.where(de["log2FC"] > 0, "tab:red", "tab:blue"), "lightgrey")
plt.scatter(de["log2FC"], -np.log10(de["p"]), c=c, s=12)
for g in hits.head(12).index:
    plt.annotate(g, (de.loc[g, "log2FC"], -np.log10(de.loc[g, "p"])), fontsize=8)
plt.axvline(LOG2FC_CUTOFF, ls="--", c="k", lw=.5); plt.axvline(-LOG2FC_CUTOFF, ls="--", c="k", lw=.5)
plt.xlabel(f"log2 fold change ({GROUP_A} - {GROUP_B})"); plt.ylabel("-log10 p")
plt.title("Volcano: red = higher in NR, blue = higher in R")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/05_volcano.png", dpi=150); plt.close()

if len(hits) >= 2:
    order = meta.sort_values(GROUP_COL).index
    zh = z.loc[hits.index[:50], order]
    sns.clustermap(zh, col_cluster=False, cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5,
                   col_colors=groups[order].map(color_map), xticklabels=False,
                   figsize=(9, max(4, 0.25 * len(zh))))
    plt.savefig(f"{OUTDIR}/06_heatmap_hits.png", dpi=150); plt.close()


# %% STEP 9 — Over-representation analysis (pathways) -------------------------
# The paper reports processes like neutrophil degranulation, cell-substrate
# adhesion, ECM organisation. Download a GMT (e.g. Reactome or GO BP from
# MSigDB / Enrichr libraries). KEY POINT: the background ("universe") must be
# the proteins you QUANTIFIED in serum, not the whole genome — otherwise
# anything 'serum-like' (complement, coagulation) looks falsely enriched.
def read_gmt(path):
    sets = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            sets[parts[0]] = set(parts[2:])
    return sets

def ora(hit_genes, universe, gene_sets, min_size=5, max_size=500):
    hit_genes, universe = set(hit_genes) & set(universe), set(universe)
    N, n, rows = len(universe), len(hit_genes), []
    for name, gs in gene_sets.items():
        gs = gs & universe
        if not (min_size <= len(gs) <= max_size):
            continue
        k = len(gs & hit_genes)
        if k == 0:
            continue
        p = stats.hypergeom.sf(k - 1, N, len(gs), n)     # P(X >= k)
        rows.append((name, k, len(gs), k / n / (len(gs) / N), p,
                     ",".join(sorted(gs & hit_genes))))
    out = pd.DataFrame(rows, columns=["term", "overlap", "set_size",
                                      "fold_enrichment", "p", "genes"])
    if len(out):
        out["FDR"] = bh_fdr(out["p"]); out = out.sort_values("p")
    return out

if os.path.exists(GMT_FILE):
    gsets = read_gmt(GMT_FILE)
    for direction in de["direction"].unique():
        sel = hits[hits["direction"] == direction].index
        enr = ora(sel, de.index, gsets)
        enr.to_csv(f"{OUTDIR}/ora_{direction}.csv", index=False)
        print(f"\nTop terms {direction}:\n", enr.head(10)[["term", "overlap", "FDR"]])
else:
    print(f"[STEP 9 skipped] put a GMT file at {GMT_FILE}")


# %% STEP 10 — Validation in an independent cohort ---------------------------
# Logic of the paper: a marker is credible if the SAME direction and
# significance replicate in independent cohorts (different centre and often a
# different MS platform). Final signature = markers significant in >= 2 cohorts.
# Because platforms differ, compare DIRECTION + p-values, never raw intensities.
def run_preprocessing(raw_df, meta_df):
    """Re-runs STEPs 2-5 on another cohort; returns imputed matrix + mask."""
    global samples, groups
    meta_df = meta_df.set_index("sample_id")
    samples = [s for s in meta_df.index if s in raw_df.columns]
    meta_df = meta_df.loc[samples]; groups = meta_df[GROUP_COL]
    lg = np.log2(clean_matrix(raw_df).replace(0, np.nan))
    k = (valid_frac(lg, GROUP_A) >= MIN_VALID_FRAC) | (valid_frac(lg, GROUP_B) >= MIN_VALID_FRAC)
    nm = median_normalise(lg.loc[k])
    return impute_downshift(nm), nm.isna(), meta_df

if DEMO:
    v_raw, v_meta = make_demo(n_R=40, n_NR=40, seed=7)
elif os.path.exists(VALIDATION_PROTEIN_FILE):
    v_raw, v_meta = load_inputs(VALIDATION_PROTEIN_FILE, VALIDATION_METADATA_FILE)
else:
    v_raw = None

if v_raw is not None:
    disc_meta, disc_samples, disc_groups = meta, samples, groups   # keep discovery
    v_imp, v_mask, v_meta = run_preprocessing(v_raw, v_meta)
    v_de = differential(v_imp, v_meta, mask=v_mask, covariates=[])
    samples, groups = disc_samples, disc_groups                      # restore
    # Only test the discovery hits -> far fewer tests, so FDR is recomputed on them
    cand = [g for g in hits.index if g in v_de.index]
    val = pd.DataFrame({
        "log2FC_disc": de.loc[cand, "log2FC"], "FDR_disc": de.loc[cand, "FDR"],
        "log2FC_val": v_de.loc[cand, "log2FC"], "p_val": v_de.loc[cand, "p"]})
    val["FDR_val_on_candidates"] = bh_fdr(val["p_val"])
    val["same_direction"] = np.sign(val["log2FC_disc"]) == np.sign(val["log2FC_val"])
    val["replicated"] = val["same_direction"] & (val["p_val"] < 0.05)
    val = val.sort_values("p_val")
    val.to_csv(f"{OUTDIR}/validation.csv")
    print(f"{val['replicated'].sum()}/{len(cand)} discovery hits replicated "
          f"(missing from validation: {len(hits) - len(cand)})")
    print(val[val["replicated"]].round(4))
    signature = val.index[val["replicated"]].tolist()
else:
    signature = hits.index.tolist()
    print("[STEP 10 skipped] no validation cohort given")


# %% STEP 11 — Literature overlap --------------------------------------------
# The paper cross-checked its markers against previously published
# melanoma / ICI signatures. Put each published list in a text file (one gene
# per line) or edit the dict below.
published = {
    # "my_lit_signature": set(open("lit_signature.txt").read().split()),
    "example_acute_phase": {"CRP", "SAA1", "SAA2", "S100A8", "S100A9", "LBP", "ORM1"},
}
universe = set(de.index)
for name, gs in published.items():
    ov = set(signature) & gs
    k, M, n, N = len(ov), len(universe), len(gs & universe), len(signature)
    p = stats.hypergeom.sf(k - 1, M, n, N) if k else 1.0
    print(f"{name}: overlap {k} ({sorted(ov)}), hypergeometric p = {p:.2g}")


# %% STEP 12 — Survival analysis in TCGA-SKCM --------------------------------
# The paper asked: are the serum markers' GENES prognostic in melanoma tumours?
# Get TCGA-SKCM RNA-seq (e.g. UCSC Xena or cBioPortal) + clinical (OS time, OS
# event). Note: this is tumour mRNA, not serum protein, and OS under mostly
# pre-immunotherapy treatment -> it's supportive evidence, not proof.
#   expression TSV: rows = genes, columns = patient IDs (log2 TPM / RSEM)
#   clinical TSV:   columns patient_id, OS_time, OS_event (1 = death)
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    HAVE_LIFELINES = True
except ImportError:
    HAVE_LIFELINES = False

if HAVE_LIFELINES and os.path.exists(TCGA_EXPR_FILE):
    expr = pd.read_csv(TCGA_EXPR_FILE, sep="\t", index_col=0)
    clin = pd.read_csv(TCGA_CLIN_FILE, sep="\t").set_index("patient_id")
    common = clin.index.intersection(expr.columns)
    expr, clin = expr[common], clin.loc[common].dropna(subset=["OS_time", "OS_event"])
    rows = []
    for g in [g for g in signature if g in expr.index]:
        x = expr.loc[g, clin.index]
        high = x > x.median()                 # median split (as in GEPIA-style analyses)
        lr = logrank_test(clin.OS_time[high], clin.OS_time[~high],
                          clin.OS_event[high], clin.OS_event[~high])
        cph = CoxPHFitter().fit(pd.DataFrame({"T": clin.OS_time, "E": clin.OS_event,
                                              "z": (x - x.mean()) / x.std()}), "T", "E")
        rows.append((g, lr.p_value, cph.hazard_ratios_["z"], cph.summary.loc["z", "p"]))
        fig, ax = plt.subplots(figsize=(5, 4))
        for lab, m in [("high", high), ("low", ~high)]:
            KaplanMeierFitter().fit(clin.OS_time[m], clin.OS_event[m], label=lab).plot(ax=ax)
        ax.set_title(f"{g}  log-rank p={lr.p_value:.3g}")
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/km_{g}.png", dpi=120); plt.close()
    surv = pd.DataFrame(rows, columns=["gene", "p_logrank", "HR_per_SD", "p_cox"])
    surv["FDR_logrank"] = bh_fdr(surv["p_logrank"])
    surv.sort_values("p_logrank").to_csv(f"{OUTDIR}/tcga_survival.csv", index=False)
    print(surv.sort_values("p_logrank").round(4))
else:
    print("[STEP 12 skipped] needs lifelines + TCGA files")


# %% STEP 13 — (optional extension) predictive model ----------------------------
# NOT in the paper's core analysis, but the natural next question: how well do
# the markers PREDICT response? Rules to avoid fooling yourself:
#   * Feature selection must happen INSIDE cross-validation (nested), otherwise
#     AUC is optimistically biased (data leakage).
#   * Best test = train on discovery, evaluate once on the validation cohort.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, RocCurveDisplay

y = (meta[GROUP_COL] == GROUP_A).astype(int).values   # 1 = NR
Xd = imp[meta.index].T
model = Pipeline([("scale", StandardScaler()),
                  ("select", SelectKBest(f_classif, k=min(10, Xd.shape[1]))),
                  ("clf", LogisticRegression(C=0.5, max_iter=1000))])
cv = StratifiedKFold(5, shuffle=True, random_state=1)
prob = cross_val_predict(model, Xd, y, cv=cv, method="predict_proba")[:, 1]
print(f"Discovery cross-validated AUC: {roc_auc_score(y, prob):.2f}")

if v_raw is not None:
    shared = [g for g in Xd.columns if g in v_imp.index]
    model.fit(Xd[shared], y)
    yv = (v_meta[GROUP_COL] == GROUP_A).astype(int).values
    pv = model.predict_proba(v_imp.loc[shared, v_meta.index].T)[:, 1]
    print(f"External validation AUC: {roc_auc_score(yv, pv):.2f}")
    RocCurveDisplay.from_predictions(yv, pv); plt.title("External validation")
    plt.savefig(f"{OUTDIR}/07_roc_validation.png", dpi=150); plt.close()

print(f"\nDone. All outputs in ./{OUTDIR}/")
