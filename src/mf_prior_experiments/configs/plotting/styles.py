X_LABEL = "Runtime [s]"
Y_LABEL = "Error"

ALGORITHMS = {
    "random_search": "RS",
    "random_search_prior": "RS+Prior",
    "random_search_prior-50": "RS+Prior(50%)",
    "grid_search": "Grid Search",
    "BOHB": "BOHB",
    "successive_halving": "SH",
    "successive_halving_prior": "SH+Prior",
    "successive_halving_prior-50": "SH+Prior(50%)",
    "hyperband": "HB",
    "hyperband_prior": "HB+Prior",
    "hyperband_prior-50": "HB+Prior(50%)",
    "asha": "ASHA",
    "asha_prior": "ASHA+Prior",
    "asha_prior-50": "ASHA+Prior(50%)",
    "asha_hyperband": "ASHA-HB",
    "asha_hyperband_prior": "ASHA-HB+Prior",
    "asha_hyperband_prior-50": "ASHA-HB+Prior(50%)",
    "ours_v1": "V1",
    "ours_v1_2": "V1_2",
    "ours_v1_3": "V1_3",
    "mfp_tpe": "MFP-TPE",
    "ours_v2": "V2",
    "ours_v2_2": "V2_2",
    "ours_v2_3": "V2_3",
    "ours_v3": "V3",
    "ours_v3_2": "V3_2",
    "ours_v4_sh": "V4_SH",
    "ours_v4_asha": "V4_ASHA",
    "ours_v4_hb": "V4_HB",
    "ours_v4_asha_hb": "V4_ASHA-HB",
    "ours_v4_v3_2": "V4_V3_2",
}

DATASETS = {
    # jahs cifar10
    "jahs_cifar10_prior-default": "CIFAR-10",
    "jahs_cifar10_prior-good": "CIFAR-10 (Good)",
    "jahs_cifar10_prior-bad": "CIFAR-10 (Bad)",
    # jahs colorectal histology
    "jahs_colorectal_histology_prior-default": "Colorectal-Histology",
    "jahs_colorectal_histology_prior-good": "Colorectal-Histology (Good)",
    "jahs_colorectal_histology_prior-bad": "Colorectal-Histology (Bad)",
    # jahs fashion_mnist
    "jahs_fashion_mnist_prior-default": "Fashion-MNIST",
    "jahs_fashion_mnist_prior-good": "Fashion-MNIST (Good)",
    "jahs_fashion_mnist_prior-bad": "Fashion-MNIST (Bad)",
    # mfh3 good
    "mfh3_good_prior-default": "Hartmann 3 (good)",
    "mfh3_good_prior-good": "H3d (good corr-good prior)",
    "mfh3_good_prior-bad": "H3d (good corr-bad prior)",
    # mfh3 moderate
    "mfh3_moderate_prior-default": "Hartmann 3 (moderate)",
    "mfh3_moderate_prior-good": "H3d (moderate corr-good prior)",
    "mfh3_moderate_prior-bad": "H3d (moderate corr-bad prior)",
    # mfh3 bad
    "mfh3_bad_rate_prior-default": "Hartmann 3 (bad)",
    "mfh3_bad_rate_prior-good": "H3d (bad corr-good prior)",
    "mfh3_bad_rate_prior-bad": "H3d (bad corr-bad prior)",
    # mfh3 terrible
    "mfh3_terrible_prior-default": "Hartmann 3 (terrible)",
    "mfh3_terrible_prior-good": "H3d (terrible corr-good prior)",
    "mfh3_terrible_prior-bad": "H3d (terrible corr-bad prior)",
    # mfh6 good
    "mfh6_good_prior-default": "Hartmann 6 (good)",
    "mfh6_good_prior-good": "H6d (good corr-good prior)",
    "mfh6_good_prior-bad": "H6d (good corr-bad prior)",
    # mfh6 moderate
    "mfh6_moderate_prior-default": "Hartmann 6 (moderate)",
    "mfh6_moderate_prior-good": "H6d (moderate corr-good prior)",
    "mfh6_moderate_prior-bad": "H6d (moderate corr-bad prior)",
    # mfh6 bad
    "mfh6_bad_prior-default": "Hartmann 6 (bad)",
    "mfh6_bad_prior-good": "H6d (bad corr-good prior)",
    "mfh6_bad_prior-bad": "H6d (bad corr-bad prior)",
    # mfh6 terrible
    "mfh6_terrible_prior-default": "Hartmann 6 (terrible)",
    "mfh6_terrible_prior-good": "H6d (terrible corr-good prior)",
    "mfh6_terrible_prior-bad": "H6d (terrible corr-bad prior)",
}

# https://matplotlib.org/stable/gallery/color/named_colors.html
COLOR_MARKER_DICT = {
    "random_search": "gray",
    "random_search_prior": "black",
    "random_search_prior-50": "darkgray",
    # "grid_search": "mediumpurple",
    # "BOHB": "lightgreen",
    "successive_halving": "navajowhite",
    "successive_halving_prior": "tan",
    "successive_halving_prior-50": "orange",
    "hyperband": "lightgreen",
    "hyperband_prior": "darkgreen",
    "hyperband_prior-50": "seagreen",
    "asha": "pink",
    "asha_prior": "magenta",
    "asha_prior-50": "palevioletred",
    "asha_hyperband": "lightyellow",
    "asha_hyperband_prior": "yellow",
    "asha_hyperband_prior-50": "khaki",
    "ours_v1": "blue",
    "ours_v1_2": "turquoise",
    "ours_v1_3": "cyan",
    "mfp_tpe": "indigo",
    "ours_v2": "olivedrab",
    "ours_v2_2": "lime",
    "ours_v2_3": "darkseagreen",
    "ours_v3": "darkslateblue",
    "ours_v3_2": "slateblue",
    "ours_v4_sh": "orangered",
    "ours_v4_asha": "darkmagenta",
    "ours_v4_hb": "limegreen",
    "ours_v4_asha_hb": "gold",
    "ours_v4_v3_2": "blueviolet",
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
