X_LABEL = "Runtime [s]"
Y_LABEL = "Error"

ALGORITHMS = {
    "random_search": "RS",
    "random_search_prior": "RS+Prior",
    "grid_search": "Grid Search",
    "BOHB": "BOHB",
    "successive_halving": "SH",
    "successive_halving_prior": "SH+Prior",
    "hyperband": "HB",
    "hyperband_prior": "HB+Prior",
    "asha": "ASHA",
    "asha_prior": "ASHA+Prior",
    "asha_hyperband": "ASHA-HB",
    "asha_hyperband_prior": "ASHA-HB+Prior",
}

DATASETS = {
    "jahs_cifar10": "CIFAR-10",
    "jahs_cifar10_good": "CIFAR-10 (Good)",
    "jahs_cifar10_bad": "CIFAR-10 (Bad)",
    "jahs_colorectal_histology": "Colorectal-Histology",
    "jahs_fashion_mnist": "Fashion-MNIST",
    "mfh3_good": "Hartmann 3 (good)",
    "mfh3_good_good": "H3d (good corr-good prior)",
    "mfh3_good_bad": "H3d (good corr-bad prior)",
    "mfh3_moderate": "Hartmann 3 (moderate)",
    "mfh3_bad": "Hartmann 3 (bad)",
    "mfh3_terrible": "Hartmann 3 (terrible)",
    "mfh3_terrible_good": "H3d (terrible corr-good prior)",
    "mfh3_terrible_bad": "H3d (terrible corr-bad prior)",
    "mfh6_good": "Hartmann 6 (good)",
    "mfh6_good_good": "H6d (good corr-good prior)",
    "mfh6_good_bad": "H6d (good corr-bad prior)",
    "mfh6_moderate": "Hartmann 6 (moderate)",
    "mfh6_bad": "Hartmann 6 (bad)",
    "mfh6_terrible": "Hartmann 6 (terrible)",
    "mfh6_terrible_good": "H6d (terrible corr-good prior)",
    "mfh6_terrible_bad": "H6d (terrible corr-bad prior)",
}

COLOR_MARKER_DICT = {
    "random_search": "black",
    "random_search_prior": "gray",
    "grid_search": "mediumpurple",
    "BOHB": "lightgreen",
    "successive_halving": "red",
    "hyperband": "green",
    "asha": "magenta",
    "successive_halving_prior": "orange",
    "hyperband_prior": "lightgreen",
    "asha_prior": "pink",
    "asha_hyperband": "yellow",
    "asha_hyperband_prior": "cyan",
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
