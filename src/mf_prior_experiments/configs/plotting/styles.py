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
    "jahs_colorectal_histology": "Colorectal-Histology",
    "jahs_fashion_mnist": "Fashion-MNIST",
    "mfh3_good": "Hartmann 3 (good)",
    "mfh3_terrible": "Hartmann 3 (terrible)",
    "mfh6_good": "Hartmann 6 (good)",
    "mfh6_terrible": "Hartmann 6 (terrible)",
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
