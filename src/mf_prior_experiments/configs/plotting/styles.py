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
    "bo": "BO(5)",
    "bo-10": "BO(10)",
    "bohb": "BOHB",
    "mobster": "MOBSTER",
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
    "pibo-default-first": "\u03C0BO(5)",
    "pibo-default-first-10": "\u03C0BO(10)",
    "asha_priorband": "ASHA-PriorBand",
    "asha_hb_priorband": "AsyncHB-PriorBand",
    # PriorBand
    "priorband": "PriorBand",
    "priorband_bo": "PriorBand(geom)+BO",
    "priorband_bo_linear": "PriorBand(HS+dyna)",
    "priorband_bo_cr_dyna": "PriorBand+BO",  #  "PriorBand-Cr-dyna",
    "priorband_bo_hs_decay": "PriorBand-HS-decay",
    "priorband_crossover_constant": "PB(Cr+const.)",
    "priorband_crossover_constant_linear": "PB(Cr+const.)",
    "priorband_crossover_decay": "PB(Cr+decay)",
    "priorband_crossover_decay_linear": "PB(Cr+decay)",
    "priorband_crossover_dynamic": "PB(Cr+dyna)",
    "priorband_crossover_dynamic_linear": "PriorBand",  # "PB(Cr+dyna+linear)",
    "priorband_hypersphere_constant": "PB(HS+const.)",
    "priorband_hypersphere_constant_linear": "PB(HS+const)",
    "priorband_hypersphere_decay": "PB(HS+decay)",
    "priorband_hypersphere_decay_linear": "PB(HS+decay)",
    "priorband_hypersphere_dynamic": "PB(HS+dyna)",
    "priorband_hypersphere_dynamic_linear": "PB(HS+dyna)",
    "priorband_noinc_prior": "HB+Prior(Geom)",  # "PB-inc(+Pr)",
    "priorband_noinc_prior_linear": "HB+Prior(linear)",
    "priorband_noinc_random": "PB-inc(+RS)",
    "priorband_noprior_inc_cr": "HB+inc(Cr)",
    "priorband_noprior_inc_cr_linear": "HB+inc(Cr)",
    "priorband_noprior_inc_hs": "HB+inc(HS)",
    "priorband_noprior_inc_hs_linear": "HB+inc(HS)",
}

DATASETS = {
    # jahs cifar10
    "jahs_cifar10_prior-good": "CIFAR-10 (ultra)",
    "jahs_cifar10_prior-medium": "CIFAR-10 (strong)",
    "jahs_cifar10_prior-at25": "CIFAR-10 (good)",
    "jahs_cifar10_prior-bad": "CIFAR-10 (bad)",
    # jahs colorectal histology
    "jahs_colorectal_histology_prior-good": "Colorectal-Histology (ultra)",
    "jahs_colorectal_histology_prior-medium": "Colorectal-Histology (strong)",
    "jahs_colorectal_histology_prior-at25": "Colorectal-Histology (good)",
    "jahs_colorectal_histology_prior-bad": "Colorectal-Histology (bad)",
    # jahs fashion_mnist
    "jahs_fashion_mnist_prior-good": "Fashion-MNIST (ultra)",
    "jahs_fashion_mnist_prior-medium": "Fashion-MNIST (strong)",
    "jahs_fashion_mnist_prior-at25": "Fashion-MNIST (good)",
    "jahs_fashion_mnist_prior-bad": "Fashion-MNIST (bad)",
    # mfh3 good
    #"mfh3_good_prior-perfect-noisy0.25": "H3d (good corr-good prior)",
    "mfh3_good_prior-good": "H3d (good corr-strong prior)",
    "mfh3_good_prior-medium": "H3d (good corr-weak prior)",
    "mfh3_good_prior-bad": "H3d (good corr-bad prior)",
    # mfh3 terrible
    #"mfh3_terrible_prior-perfect-noisy0.25": "H3d (bad corr-good prior)",
    "mfh3_terrible_prior-good": "H3d (bad corr-strong prior)",
    "mfh3_terrible_prior-medium": "H3d (bad corr-weak prior)",
    "mfh3_terrible_prior-bad": "H3d (bad corr-bad prior)",
    # mfh6 good
    #"mfh6_good_prior-perfect-noisy0.25": "H6d (good corr-good prior)",
    "mfh6_good_prior-good": "H6d (good corr-strong prior)",
    "mfh6_good_prior-medium": "H6d (good corr-weak prior)",
    "mfh6_good_prior-bad": "H6d (good corr-bad prior)",
    # mfh6 terrible
    #"mfh6_terrible_prior-perfect-noisy0.25": "H6d (bad corr-good prior)",
    "mfh6_terrible_prior-good": "H6d (bad corr-strong prior)",
    "mfh6_terrible_prior-medium": "H6d (bad corr-weak prior)",
    "mfh6_terrible_prior-bad": "H6d (bad corr-bad prior)",
    # lcbench-126026
    "lcbench-126026_prior-bad": "LCBench-126026 (bad)",
    "lcbench-126026_prior-medium": "LCBench-126026 (strong)",
    "lcbench-126026_prior-at25": "LCBench-126026 (good)",
    "lcbench-126026_prior-good": "LCBench-126026 (ultra)",
    # lcbench-167190
    "lcbench-167190_prior-bad": "LCBench-167190 (bad)",
    "lcbench-167190_prior-medium": "LCBench-167190 (strong)",
    "lcbench-167190_prior-at25": "LCBench-167190 (good)",
    "lcbench-167190_prior-good": "LCBench-167190 (ultra)",
    # lcbench-168330
    "lcbench-168330_prior-bad": "LCBench-168330 (bad)",
    "lcbench-168330_prior-medium": "LCBench-168330 (strong)",
    "lcbench-168330_prior-at25": "LCBench-168330 (good)",
    "lcbench-168330_prior-good": "LCBench-168330 (ultra)",
    # lcbench-168910
    "lcbench-168910_prior-bad": "LCBench-168910 (bad)",
    "lcbench-168910_prior-medium": "LCBench-168910 (strong)",
    "lcbench-168910_prior-at25": "LCBench-168910 (good)",
    "lcbench-168910_prior-good": "LCBench-168910 (ultra)",
    # lcbench-189906
    "lcbench-189906_prior-bad": "LCBench-189906 (bad)",
    "lcbench-189906_prior-medium": "LCBench-189906 (strong)",
    "lcbench-189906_prior-at25": "LCBench-189906 (good)",
    "lcbench-189906_prior-good": "LCBench-189906 (ultra)",
    # translate wmt
    "translatewmt_xformer_64_prior-bad": "PD1-WMT (bad)",
    "translatewmt_xformer_64_prior-medium": "PD1-WMT (strong)",
    "translatewmt_xformer_64_prior-at25": "PD1-WMT (good)",
    "translatewmt_xformer_64_prior-good": "PD1-WMT (ultra)",
    # lm1b
    "lm1b_transformer_2048_prior-bad": "PD1-Lm1b (bad)",
    "lm1b_transformer_2048_prior-at25": "PD1-Lm1b (good)",
    "lm1b_transformer_2048_prior-medium": "PD1-Lm1b (strong)",
    "lm1b_transformer_2048_prior-good": "PD1-Lm1b (ultra)",
    # uniref
    "uniref50_transformer_prior-bad": "PD1-Uniref50 (bad)",
    "uniref50_transformer_prior-at25": "PD1-Uniref50 (good)",
    "uniref50_transformer_prior-medium": "PD1-Uniref50 (strong)",
    "uniref50_transformer_prior-good": "PD1-Uniref50 (ultra)",
    # imagenet
    "imagenet_resnet_512_prior-bad": "Imagenet (bad)",
    "imagenet_resnet_512_prior-at25": "Imagenet (good)",
    "imagenet_resnet_512_prior-medium": "Imagenet (strong)",
    "imagenet_resnet_512_prior-good": "Imagenet (ultra)",
    # cifar100
    "cifar100_wideresnet_2048_prior-bad": "CIFAR100 (bad)",
    "cifar100_wideresnet_2048_prior-at25": "CIFAR100 (good)",
    "cifar100_wideresnet_2048_prior-medium": "CIFAR100 (strong)",
    "cifar100_wideresnet_2048_prior-good": "CIFAR100 (ultra)",
}

# https://matplotlib.org/stable/gallery/color/named_colors.html
COLOR_MARKER_DICT = {
    "random_search": "palevioletred",
    "random_search_prior": "magenta",
    "random_search_prior-50": "indigo",
    "bo": "darkgoldenrod",
    "bo-10": "darkgoldenrod",
    "bohb": "darkcyan",
    "mobster": "black",
    # "successive_halving": "SH",
    # "successive_halving_prior": "SH+Prior",
    # "successive_halving_prior-50": "SH+Prior(50%)",
    "hpbandster": "darkgreen",
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
    "pibo-default-first-10": "firebrick",
    "asha_priorband": "palevioletred",
    "asha_hb_priorband": "darkmagenta",
    # PriorBand
    "priorband": "blue",
    "priorband_bo": "gray",
    "priorband_bo_linear": "steelblue",
    "priorband_bo_cr_dyna": "seagreen",
    "priorband_bo_hs_decay": "palevioletred",
    "priorband_crossover_constant": "olivedrab",
    "priorband_crossover_constant_linear": "green",
    "priorband_crossover_decay": "blue",
    "priorband_crossover_decay_linear": "skyblue",
    "priorband_crossover_dynamic": "brown",
    "priorband_crossover_dynamic_linear": "blue",
    "priorband_hypersphere_constant": "pink",
    "priorband_hypersphere_constant_linear": "orange",
    "priorband_hypersphere_decay": "turquoise",
    "priorband_hypersphere_decay_linear": "gold",
    "priorband_hypersphere_dynamic": "red",
    "priorband_hypersphere_dynamic_linear": "violet",
    "priorband_noinc_prior": "chocolate",  # "PB-inc(+Pr)",
    "priorband_noinc_prior_linear": "green",
    "priorband_noinc_random": "turquoise",
    "priorband_noprior_inc_cr": "orange",
    "priorband_noprior_inc_cr_linear": "red",
    "priorband_noprior_inc_hs": "seagreen",
    "priorband_noprior_inc_hs_linear": "skyblue",
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
