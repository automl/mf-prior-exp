X_LABEL = "Approx. Full Evaluations"
Y_LABEL = "Validation Error [%]"

ALGORITHMS = {
    "random_search": "Random Search",
    "grid_search": "Grid Search",
    "BOHB": "BOHB"

}

DATASETS = {
    "jahs_cifar10": "CIFAR-10",
    "jahs_colorectal_histology": "Colorectal-Histology",
    "jahs_fashion_mnist": "Fashion-MNIST"
}


COLOR_MARKER_DICT = {
    'random_search': "black",
    "grid_search": "mediumpurple",
    "BOHB": "lightgreen",
}

Y_MAP = {
    "jahs_cifar10": [0, 100],
    "jahs_colorectal_histology": [0, 100],
    "jahs_fashion_mnist": [0, 100],
}

X_MAP = [0, 25, 50, 75, 100]

WIDTH_PT = 398.33864
