X_LABEL = dict(
    {
        False: "Full trainings",  # if `cost_as_runtime` is False
        True: "Runtime [s]",
    }
)

# "Runtime [s]"
Y_LABEL = "Error"

ALGORITHMS = {
    "random_search": "RS",
    "random_search_prior": "RS+Prior",
    "random_search_prior-50": "RS+Prior(50%)",
    "bo": "BO",
    "bohb": "BOHB",
    "successive_halving": "SH",
    "successive_halving_prior": "SH+Prior",
    "successive_halving_prior-50": "SH+Prior(50%)",
    "hpbandster": "HpBandSter",
    "hyperband": "HB",
    "hyperband_prior": "HB+Prior",
    "hyperband_prior-50": "HB+Prior(50%)",
    "asha": "ASHA",
    "asha_prior": "ASHA+Prior",
    "asha_prior-50": "ASHA+Prior(50%)",
    "asha_hyperband": "AsyncHB",
    "asha_hyperband_prior": "AAsyncHB+Prior",
    "asha_hyperband_prior-50": "AsyncHB+Prior(50%)",
    "pibo": "\u03C0BO",
    "pibo-default-first": "\u03C0BO",
    "asha_priorband": "ASHA-PriorBand",
    "asha_hb_priorband": "AsyncHB-PriorBand",
    # PriorBand
    "priorband": "PriorBand",
    "priorband_crossover_constant": "PB(Cr+const.)",
    "priorband_crossover_decay": "PB(Cr+decay)",
    "priorband_crossover_dynamic": "PB(Cr+const.)",
    "priorband_hypersphere_constant": "PB(HS+const.)",
    "priorband_hypersphere_decay": "PB(HS+decay)",
    "priorband_hypersphere_dynamic": "PB(HS+dyna)",
    "priorband_noinc_prior": "HB+Prior(Geom)",  # "PB-inc(+Pr)",
    "priorband_noinc_random": "PB-inc(+RS)",
    "priorband_noprior_inc_cr": "HB+inc(Cr)",
    "priorband_noprior_inc_hs": "HB+inc(HS)",
}

DATASETS = {
    # jahs cifar10
    "jahs_cifar10_prior-good": "CIFAR-10 (good)",
    "jahs_cifar10_prior-bad": "CIFAR-10 (bad)",
    # jahs colorectal histology
    "jahs_colorectal_histology_prior-good": "Colorectal-Histology (good)",
    "jahs_colorectal_histology_prior-bad": "Colorectal-Histology (bad)",
    # jahs fashion_mnist
    "jahs_fashion_mnist_prior-good": "Fashion-MNIST (good)",
    "jahs_fashion_mnist_prior-bad": "Fashion-MNIST (bad)",
    # mfh3 good
    "mfh3_good_prior-perfect-noisy0.25": "H3d (good corr-good prior)",
    "mfh3_good_prior-bad": "H3d (good corr-bad prior)",
    # mfh3 terrible
    "mfh3_terrible_prior-perfect-noisy0.25": "H3d (bad corr-good prior)",
    "mfh3_terrible_prior-bad": "H3d (bad corr-bad prior)",
    # mfh6 good
    "mfh6_good_prior-perfect-noisy0.25": "H6d (good corr-good prior)",
    "mfh6_good_prior-bad": "H6d (good corr-bad prior)",
    # mfh6 terrible
    "mfh6_terrible_prior-perfect-noisy0.25": "H6d (bad corr-good prior)",
    "mfh6_terrible_prior-bad": "H6d (terrible corr-bad prior)",
    # lcbench-126026
    "lcbench-126026_prior-bad": "LCBench-126026 (bad)",
    "lcbench-126026_prior-good": "LCBench-126026 (good)",
    # lcbench-167190
    "lcbench-167190_prior-bad": "LCBench-167190 (bad)",
    "lcbench-167190_prior-good": "LCBench-167190 (good)",
    # lcbench-168330
    "lcbench-168330_prior-bad": "LCBench-168330 (bad)",
    "lcbench-168330_prior-good": "LCBench-168330 (good)",
    # lcbench-168910
    "lcbench-168910_prior-bad": "LCBench-168910 (bad)",
    "lcbench-168910_prior-good": "LCBench-168910 (good)",
    # lcbench-189906
    "lcbench-189906_prior-bad": "LCBench-189906 (bad)",
    "lcbench-189906_prior-good": "LCBench-189906 (good)",
    # translate wmt
    "translatewmt_xformer_64_prior-bad": "PD1-WMT (bad)",
    "translatewmt_xformer_64_prior-good": "PD1-WMT (good)",
    # lm1b
    "lm1b_transformer_2048_prior-bad": "PD1-Lm1b (bad)",
    "lm1b_transformer_2048_prior-good": "PD1-Lm1b (good)",
    # uniref
    "uniref50_transformer_prior-bad": "PD1-Uniref50 (bad)",
    "uniref50_transformer_prior-good": "PD1-Uniref50 (good)",
}

# https://matplotlib.org/stable/gallery/color/named_colors.html
COLOR_MARKER_DICT = {
    "random_search": "palevioletred",
    "random_search_prior": "magenta",
    "random_search_prior-50": "indigo",
    "bo": "darkgoldenrod",
    "bohb": "darkcyan",
    # "successive_halving": "SH",
    # "successive_halving_prior": "SH+Prior",
    # "successive_halving_prior-50": "SH+Prior(50%)",
    # "hpbandster": "HpBandSter",
    "hyperband": "limegreen",
    "hyperband_prior": "darkgreen",
    "hyperband_prior-50": "olive",
    "asha": "pink",
    "asha_prior": "magenta",
    "asha_prior-50": "pink",
    "asha_hyperband": "peru",
    "asha_hyperband_prior": "olive",
    "asha_hyperband_prior-50": "peru",
    "pibo": "firebrick",
    "pibo-default-first": "firebrick",
    "asha_priorband": "palevioletred",
    "asha_hb_priorband": "darkmagenta",
    # PriorBand
    "priorband": "blue",
    "priorband_crossover_constant": "olivedrab",
    "priorband_crossover_decay": "blue",
    "priorband_crossover_dynamic": "brown",
    "priorband_hypersphere_constant": "brown",
    "priorband_hypersphere_decay": "olivedrab",
    "priorband_hypersphere_dynamic": "red",
    "priorband_noinc_prior": "chocolate",  # "PB-inc(+Pr)",
    "priorband_noinc_random": "turquoise",
    "priorband_noprior_inc_cr": "orange",
    "priorband_noprior_inc_hs": "seagreen",
}

Y_MAP = {
    "jahs_cifar10": [0, 100],
    "jahs_colorectal_histology": [0, 100],
    "jahs_fashion_mnist": [0, 100],
    "mfh3_good": [-10, 10],
    "mfh6_good": [-10, 10],
}

X_MAP = [0, 25, 50, 75, 100]

WIDTH_PT = 398.33864
