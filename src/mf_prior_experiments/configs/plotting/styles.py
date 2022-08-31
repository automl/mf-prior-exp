X_LABEL = "Runtime [s]"
Y_LABEL = "Error"

ALGORITHMS = {
    "random_search": "Random Search",
    "grid_search": "Grid Search",
    "BOHB": "BOHB",
    "sh": "SH",
    "sh_prior": "SH+Prior",
    "hb": "HB",
    "hb_prior": "HB+Prior",
    "asha": "ASHA",
    "asha_prior": "ASHA+Prior",
}

DATASETS = {
    "jahs_cifar10": "CIFAR-10",
    "jahs_colorectal_histology": "Colorectal-Histology",
    "jahs_fashion_mnist": "Fashion-MNIST",
    "mfh3_good": "Hartmann 3 (good)",
    "mfh6_good": "Hartmann 6 (good)",
}

COLOR_MARKER_DICT = {
    "random_search": "black",
    "grid_search": "mediumpurple",
    "BOHB": "lightgreen",
    "sh": "red",
    "hb": "green",
    "asha": "magenta",
    "sh_prior": "orange",
    "hb_prior": "lightgreen",
    "asha_prior": "pink",
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
