from __future__ import annotations

X_LABEL = {
    "cumulated_fidelity": "Full trainings",
    "end_time_since_global_start": "Approx. full trainings",
}

BENCH_TABLE_NAMES = {
    "cifar100_wideresnet_2048_prior-at25": "PD1-Cifar100 (good)",
    "cifar100_wideresnet_2048_prior-bad": "PD1-Cifar100 (bad)",
    "cifar100_wideresnet_2048_prior-good": "PD1-Cifar100 (ultra)",
    "cifar100_wideresnet_2048_prior-medium": "PD1-Cifar100 (~opt)",
    "imagenet_resnet_512_prior-at25": "PD1-ImageNet (good)",
    "imagenet_resnet_512_prior-bad": "PD1-ImageNet (bad)",
    "imagenet_resnet_512_prior-good": "PD1-ImageNet (ultra)",
    "imagenet_resnet_512_prior-medium": "PD1-ImageNet (~opt)",
    "jahs_cifar10_prior-at25": "JAHS-C10 (good)",
    "jahs_cifar10_prior-bad": "JAHS-C10 (bad)",
    "jahs_cifar10_prior-good": "JAHS-C10 (ultra)",
    "jahs_cifar10_prior-medium": "JAHS-C10 (~opt)",
    "jahs_colorectal_histology_prior-at25": "JAHS-CH (good)",
    "jahs_colorectal_histology_prior-bad": "JAHS-CH (bad)",
    "jahs_colorectal_histology_prior-good": "JAHS-CH (ultra)",
    "jahs_colorectal_histology_prior-medium": "JAHS-CH (~opt)",
    "jahs_fashion_mnist_prior-at25": "JAHS-FM (good)",
    "jahs_fashion_mnist_prior-bad": "JAHS-FM (bad)",
    "jahs_fashion_mnist_prior-good": "JAHS-FM (ultra)",
    "jahs_fashion_mnist_prior-medium": "JAHS-FM (~opt)",
    "lcbench-126026_prior-at25": "LC-126026 (good)",
    "lcbench-126026_prior-bad": "LC-126026 (bad)",
    "lcbench-126026_prior-good": "LC-126026 (ultra)",
    "lcbench-126026_prior-medium": "LC-126026 (~opt)",
    "lcbench-167190_prior-at25": "LC-167190 (good)",
    "lcbench-167190_prior-bad": "LC-167190 (bad)",
    "lcbench-167190_prior-good": "LC-167190 (ultra)",
    "lcbench-167190_prior-medium": "LC-167190 (~opt)",
    "lcbench-168330_prior-at25": "LC-168330 (good)",
    "lcbench-168330_prior-bad": "LC-168330 (bad)",
    "lcbench-168330_prior-good": "LC-168330 (ultra)",
    "lcbench-168330_prior-medium": "LC-168330 (~opt)",
    "lcbench-168910_prior-at25": "LC-168910 (good)",
    "lcbench-168910_prior-bad": "LC-168910 (bad)",
    "lcbench-168910_prior-good": "LC-168910 (ultra)",
    "lcbench-168910_prior-medium": "LC-168910 (~opt)",
    "lcbench-189906_prior-at25": "LC-189906 (good)",
    "lcbench-189906_prior-bad": "LC-189906 (bad)",
    "lcbench-189906_prior-good": "LC-189906 (ultra)",
    "lcbench-189906_prior-medium": "LC-189906 (~opt)",
    "lm1b_transformer_2048_prior-at25": "PD1-LM1B (good)",
    "lm1b_transformer_2048_prior-bad": "PD1-LM1B (bad)",
    "lm1b_transformer_2048_prior-good": "PD1-LM1B (ultra)",
    "lm1b_transformer_2048_prior-medium": "PD1-LM1B (~opt)",
    "mfh3_good_prior-at25": "MFH3-good (good)",
    "mfh3_good_prior-bad": "MFH3-good (bad)",
    "mfh3_good_prior-good": "MFH3-good (ultra)",
    "mfh3_good_prior-medium": "MFH3-good (~opt)",
    "mfh3_terrible_prior-at25": "MFH3-terrible (good)",
    "mfh3_terrible_prior-bad": "MFH3-terrible (bad)",
    "mfh3_terrible_prior-good": "MFH3-terrible (ultra)",
    "mfh3_terrible_prior-medium": "MFH3-terrible (~opt)",
    "mfh6_good_prior-at25": "MFH6-good (good)",
    "mfh6_good_prior-bad": "MFH6-good (bad)",
    "mfh6_good_prior-good": "MFH6-good (ultra)",
    "mfh6_good_prior-medium": "MFH6-good (~opt)",
    "mfh6_terrible_prior-at25": "MFH6-terrible (good)",
    "mfh6_terrible_prior-bad": "MFH6-terrible (bad)",
    "mfh6_terrible_prior-good": "MFH6-terrible (ultra)",
    "mfh6_terrible_prior-medium": "MFH6-terrible (~opt)",
    "translatewmt_xformer_64_prior-at25": "PD1-WMT (good)",
    "translatewmt_xformer_64_prior-bad": "PD1-WMT (bad)",
    "translatewmt_xformer_64_prior-good": "PD1-WMT (ultra)",
    "translatewmt_xformer_64_prior-medium": "PD1-WMT (~opt)",
}


def get_xticks(xrange: tuple[int, int], ticks: int = 5) -> list[int]:
    predefined: dict[tuple[int, int], list[int]] = {
        (0, 12): [0, 3, 6, 9, 12],
        (1, 12): [1, 3, 6, 9, 12],
        (0, 5): [0, 1, 2, 3, 4, 5],
        (1, 5): [1, 2, 3, 4, 5],
        (0, 20): [0, 5, 10, 15, 20],
        (1, 20): [1, 5, 10, 15, 20],
        (1, 10): [1, 3, 5, 7, 10]
    }
    xticks = predefined.get(xrange)
    if xticks is None:
        import numpy as np

        low, high = xrange
        return np.linspace(low, high, num=ticks, endpoint=True, dtype=int).tolist()
    else:
        return xticks


Y_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "cifar100_wideresnet_2048_prior-at25": (None, 0.275),
    "cifar100_wideresnet_2048_prior-bad": (None, 0.275),
    "cifar100_wideresnet_2048_prior-good": (None, 0.275),
    "cifar100_wideresnet_2048_prior-medium": (None, 0.275),
    "cifar100_wideresnet_2048_prior-gpt": (0.2, 0.45),
    "imagenet_resnet_512_prior-at25": (None, 0.3),
    "imagenet_resnet_512_prior-bad": (None, 0.3),
    "imagenet_resnet_512_prior-good": (None, 0.3),
    "imagenet_resnet_512_prior-medium": (None, 0.3),
    "imagenet_resnet_512_prior-gpt": (0.2, 0.4),
    "jahs_cifar10_prior-at25": (None, 12),
    "jahs_cifar10_prior-bad": (None, 12),
    "jahs_cifar10_prior-good": (None, 12),
    "jahs_cifar10_prior-medium": (None, 12),
    "jahs_colorectal_histology_prior-at25": (None, 8),
    "jahs_colorectal_histology_prior-bad": (None, 8),
    "jahs_colorectal_histology_prior-good": (None, 8),
    "jahs_colorectal_histology_prior-medium": (None, 8),
    "jahs_fashion_mnist_prior-at25": (None, 7),
    "jahs_fashion_mnist_prior-bad": (None, 7),
    "jahs_fashion_mnist_prior-good": (None, 7),
    "jahs_fashion_mnist_prior-medium": (None, 7),
    "lcbench-126026_prior-at25": (None, 0.08),
    "lcbench-126026_prior-bad": (None, 0.08),
    "lcbench-126026_prior-good": (None, 0.08),
    "lcbench-126026_prior-medium": (None, 0.08),
    "lcbench-126026_prior-gpt": (0.02, 0.06),
    "lcbench-167190_prior-at25": (None, 0.25),
    "lcbench-167190_prior-bad": (None, 0.3),
    "lcbench-167190_prior-good": (None, 0.25),
    "lcbench-167190_prior-medium": (None, 0.25),
    "lcbench-168330_prior-at25": (None, 0.5),
    "lcbench-168330_prior-bad": (None, 0.5),
    "lcbench-168330_prior-good": (None, 0.5),
    "lcbench-168330_prior-medium": (None, 0.5),
    "lcbench-168330_prior-gpt": (0.3, 0.45),
    "lcbench-168910_prior-at25": (None, 0.27),
    "lcbench-168910_prior-bad": (None, 0.27),
    "lcbench-168910_prior-good": (None, 0.27),
    "lcbench-168910_prior-medium": (None, 0.27),
    "lcbench-168910_prior-gpt": (0.2, 0.4),
    "lcbench-189906_prior-at25": (None, 0.2),
    "lcbench-189906_prior-bad": (None, 0.2),
    "lcbench-189906_prior-good": (None, 0.2),
    "lcbench-189906_prior-medium": (None, 0.2),
    "lcbench-189906_prior-gpt": (0.1, 0.2),
    "lm1b_transformer_2048_prior-at25": (None, 0.68),
    "lm1b_transformer_2048_prior-bad": (None, 0.68),
    "lm1b_transformer_2048_prior-good": (None, 0.68),
    "lm1b_transformer_2048_prior-medium": (None, 0.68),
    "lm1b_transformer_2048_prior-gpt": (0.6, 0.72),
    "mfh3_good_prior-at25": (-5, 1),
    "mfh3_good_prior-bad": (-5, 1),
    "mfh3_good_prior-good": (-5, 1),
    "mfh3_good_prior-medium": (-5, 1),
    "mfh3_terrible_prior-at25": (-5, 1),
    "mfh3_terrible_prior-bad": (-5, 1),
    "mfh3_terrible_prior-good": (-5, 1),
    "mfh3_terrible_prior-medium": (-5, 1),
    "mfh6_good_prior-at25": (None, 0),
    "mfh6_good_prior-bad": (None, 0),
    "mfh6_good_prior-good": (None, 0),
    "mfh6_good_prior-medium": (None, 0),
    "mfh6_terrible_prior-at25": (None, 0),
    "mfh6_terrible_prior-bad": (None, 0),
    "mfh6_terrible_prior-good": (None, 0),
    "mfh6_terrible_prior-medium": (None, 0),
    "translatewmt_xformer_64_prior-at25": (None, 0.42),
    "translatewmt_xformer_64_prior-bad": (None, 0.42),
    "translatewmt_xformer_64_prior-good": (None, 0.42),
    "translatewmt_xformer_64_prior-medium": (None, 0.42),
    "translatewmt_xformer_64_prior-gpt": (0.3, 0.5),
}

# "Runtime [s]"
Y_LABEL = "Error"
GOOD_CORR_COLOR = "black"  # "green"
BAD_CORR_COLOR = "black"  # "red"
BENCHMARK_COLORS = {
    # high corr
    "translatewmt_xformer_64_prior-at25": GOOD_CORR_COLOR,
    "translatewmt_xformer_64_prior-bad": GOOD_CORR_COLOR,
    "lm1b_transformer_2048_prior-at25": GOOD_CORR_COLOR,
    "lm1b_transformer_2048_prior-bad": GOOD_CORR_COLOR,
    "jahs_colorectal_histology_prior-at25": GOOD_CORR_COLOR,
    "jahs_colorectal_histology_prior-bad": GOOD_CORR_COLOR,
    "lcbench-167190_prior-at25": GOOD_CORR_COLOR,
    "lcbench-167190_prior-bad": GOOD_CORR_COLOR,
    "lcbench-168330_prior-at25": GOOD_CORR_COLOR,
    "lcbench-168330_prior-bad": GOOD_CORR_COLOR,
    "lcbench-168910_prior-at25": GOOD_CORR_COLOR,
    "lcbench-168910_prior-bad": GOOD_CORR_COLOR,
    "lcbench-189906_prior-at25": GOOD_CORR_COLOR,
    "lcbench-189906_prior-bad": GOOD_CORR_COLOR,
    "mfh3_good_prior-good": GOOD_CORR_COLOR,
    "mfh3_good_prior-bad": GOOD_CORR_COLOR,
    "mfh6_good_prior-good": GOOD_CORR_COLOR,
    "mfh6_good_prior-bad": GOOD_CORR_COLOR,
    # low corr
    "cifar100_wideresnet_2048_prior-at25": BAD_CORR_COLOR,
    "cifar100_wideresnet_2048_prior-bad": BAD_CORR_COLOR,
    "imagenet_resnet_512_prior-at25": BAD_CORR_COLOR,
    "imagenet_resnet_512_prior-bad": BAD_CORR_COLOR,
    "jahs_cifar10_prior-at25": BAD_CORR_COLOR,
    "jahs_cifar10_prior-bad": BAD_CORR_COLOR,
    "jahs_fashion_mnist_prior-at25": BAD_CORR_COLOR,
    "jahs_fashion_mnist_prior-bad": BAD_CORR_COLOR,
    "lcbench-126026_prior-good": BAD_CORR_COLOR,
    "lcbench-126026_prior-at25": BAD_CORR_COLOR,
    "lcbench-126026_prior-bad": BAD_CORR_COLOR,
    "mfh3_terrible_prior-good": BAD_CORR_COLOR,
    "mfh3_terrible_prior-bad": BAD_CORR_COLOR,
    "mfh6_terrible_prior-good": BAD_CORR_COLOR,
    "mfh6_terrible_prior-bad": BAD_CORR_COLOR,
}

DATASETS = {
    # jahs cifar10
    "jahs_cifar10_prior-good": "[-]CIFAR-10 (ultra)",
    "jahs_cifar10_prior-medium": "[-]CIFAR-10 (~opt)",
    "jahs_cifar10_prior-at25": "[-]CIFAR-10 (good)",
    "jahs_cifar10_prior-bad": "[-]CIFAR-10 (bad)",
    # jahs colorectal histology
    "jahs_colorectal_histology_prior-good": "[-]Colorectal-Histology (ultra)",
    "jahs_colorectal_histology_prior-medium": "[-]Colorectal-Histology (~opt)",
    "jahs_colorectal_histology_prior-at25": "[-]Colorectal-Histology (good)",
    "jahs_colorectal_histology_prior-bad": "[-]Colorectal-Histology (bad)",
    # jahs fashion_mnist
    "jahs_fashion_mnist_prior-good": "[-]Fashion-MNIST (ultra)",
    "jahs_fashion_mnist_prior-medium": "[-]Fashion-MNIST (~opt)",
    "jahs_fashion_mnist_prior-at25": "[-]Fashion-MNIST (good)",
    "jahs_fashion_mnist_prior-bad": "[-]Fashion-MNIST (bad)",
    # mfh3 good
    # "mfh3_good_prior-perfect-noisy0.25": "H3d (high corr-good prior)",
    "mfh3_good_prior-good": "H3d (high corr-good prior)",
    "mfh3_good_prior-medium": "H3d (high corr-weak prior)",
    "mfh3_good_prior-bad": "H3d (high corr-bad prior)",
    # mfh3 terrible
    # "mfh3_terrible_prior-perfect-noisy0.25": "H3d (low corr-good prior)",
    "mfh3_terrible_prior-good": "H3d (low corr-good prior)",
    "mfh3_terrible_prior-medium": "H3d (low corr-weak prior)",
    "mfh3_terrible_prior-bad": "H3d (low corr-bad prior)",
    # mfh6 good
    # "mfh6_good_prior-perfect-noisy0.25": "H6d (high corr-good prior)",
    "mfh6_good_prior-good": "H6d (high corr-good prior)",
    "mfh6_good_prior-medium": "H6d (high corr-weak prior)",
    "mfh6_good_prior-bad": "H6d (high corr-bad prior)",
    # mfh6 terrible
    # "mfh6_terrible_prior-perfect-noisy0.25": "H6d (low corr-good prior)",
    "mfh6_terrible_prior-good": "H6d (low corr-good prior)",
    "mfh6_terrible_prior-medium": "H6d (low corr-weak prior)",
    "mfh6_terrible_prior-bad": "H6d (low corr-bad prior)",
    # lcbench-126026
    "lcbench-126026_prior-bad": "[-]LCBench-126026 (bad)",
    "lcbench-126026_prior-medium": "[-]LCBench-126026 (~opt)",
    "lcbench-126026_prior-at25": "[-]LCBench-126026 (good)",
    "lcbench-126026_prior-good": "[-]LCBench-126026 (ultra)",
    # lcbench-167190
    "lcbench-167190_prior-bad": "[+]LCBench-167190 (bad)",
    "lcbench-167190_prior-medium": "[+]LCBench-167190 (~opt)",
    "lcbench-167190_prior-at25": "[+]LCBench-167190 (good)",
    "lcbench-167190_prior-good": "[+]LCBench-167190 (ultra)",
    # lcbench-168330
    "lcbench-168330_prior-bad": "[+]LCBench-168330 (bad)",
    "lcbench-168330_prior-medium": "[+]LCBench-168330 (~opt)",
    "lcbench-168330_prior-at25": "[+]LCBench-168330 (good)",
    "lcbench-168330_prior-good": "[+]LCBench-168330 (ultra)",
    # lcbench-168910
    "lcbench-168910_prior-bad": "[+]LCBench-168910 (bad)",
    "lcbench-168910_prior-medium": "[+]LCBench-168910 (~opt)",
    "lcbench-168910_prior-at25": "[+]LCBench-168910 (good)",
    "lcbench-168910_prior-good": "[+]LCBench-168910 (ultra)",
    # lcbench-189906
    "lcbench-189906_prior-bad": "[+]LCBench-189906 (bad)",
    "lcbench-189906_prior-medium": "[+]LCBench-189906 (~opt)",
    "lcbench-189906_prior-at25": "[+]LCBench-189906 (good)",
    "lcbench-189906_prior-good": "[+]LCBench-189906 (ultra)",
    # translate wmt
    "translatewmt_xformer_64_prior-bad": "[+]PD1-WMT (bad)",
    "translatewmt_xformer_64_prior-medium": "[+]PD1-WMT (~opt)",
    "translatewmt_xformer_64_prior-at25": "[+]PD1-WMT (good)",
    "translatewmt_xformer_64_prior-good": "[+]PD1-WMT (ultra)",
    # lm1b
    "lm1b_transformer_2048_prior-bad": "[+]PD1-Lm1b (bad)",
    "lm1b_transformer_2048_prior-at25": "[+]PD1-Lm1b (good)",
    "lm1b_transformer_2048_prior-medium": "[+]PD1-Lm1b (~opt)",
    "lm1b_transformer_2048_prior-good": "[+]PD1-Lm1b (ultra)",
    # uniref
    "uniref50_transformer_prior-bad": "PD1-Uniref50 (bad)",
    "uniref50_transformer_prior-at25": "PD1-Uniref50 (good)",
    "uniref50_transformer_prior-medium": "PD1-Uniref50 (~opt)",
    "uniref50_transformer_prior-good": "PD1-Uniref50 (ultra)",
    # imagenet
    "imagenet_resnet_512_prior-bad": "[-]Imagenet (bad)",
    "imagenet_resnet_512_prior-at25": "[-]Imagenet (good)",
    "imagenet_resnet_512_prior-medium": "[-]Imagenet (~opt)",
    "imagenet_resnet_512_prior-good": "[-]Imagenet (ultra)",
    # cifar100
    "cifar100_wideresnet_2048_prior-bad": "[-]CIFAR100 (bad)",
    "cifar100_wideresnet_2048_prior-at25": "[-]CIFAR100 (good)",
    "cifar100_wideresnet_2048_prior-medium": "[-]CIFAR100 (~opt)",
    "cifar100_wideresnet_2048_prior-good": "[-]CIFAR100 (ultra)",
}

ALGORITHMS = {
    "random_search": "RS",
    "random_search_prior": "RS+Prior",
    "random_search_prior-no-default": "RS+Prior(No-Mode)",
    "random_search_prior-50": "RS+Prior(50%)",
    "bo": "BO(5)",
    "bo-10": "BO(10)",
    "bohb": "BOHB",
    "mobster": "Mobster",
    "successive_halving": "SH",
    "successive_halving_prior": "SH+Prior",
    "successive_halving_prior-50": "SH+Prior(50%)",
    "hpbandster": "HpBandSter",
    "hyperband": "HB",
    "hyperband_prior": "HB+Prior",
    "hyperband_prior-50": "HB+Prior(50%)",
    "hyperband_prior-no-default": "HB+Prior(No-Mode)",
    "hyperband_prior-default-at-target": "HB+Prior(mode@target)",
    "asha": "ASHA",
    "asha_prior": "ASHA+Prior",
    "asha_prior-default-at-target": "ASHA+Prior",
    "asha_prior-50": "ASHA+Prior(50%)",
    "asha_hyperband": "AsyncHB",
    "asha_hyperband_prior": "AsyncHB+Prior",
    "asha_hyperband_prior-default-at-target": "AsyncHB+Prior",
    "asha_hyperband_prior-50": "AsyncHB+Prior(50%)",
    "asha_hb_priorband_bo_rung": "Mobster+PriorBand",
    "asha_hb_priorband_bo_rung-50": "AsyncHB+PB(50%)(BO)[r]",
    "asha_hb_priorband_bo_joint": "Mobster+PriorBand(joint)",
    "asha_hb_priorband_bo_joint-50": "AsyncHB+PB(50%)(BO)[j]",
    "pibo": "\u03C0BO",
    "pibo-no-default": "\u03C0BO(10)(No-Mode)",
    "pibo-default-first": "\u03C0BO(5)",
    "pibo-default-first-10": "\u03C0BO(10)",
    "asha_priorband": "ASHA-PriorBand",
    "asha_hb_priorband": "AsyncHB-PriorBand",
    "asha_hb_priorband-50": "AsyncHB-PriorBand(50%)",
    "hb_inc": "HB+inc",
    "hb_inc-50": "HB+inc(50%)",
    # PriorBand
    "priorband": "PriorBand",
    "priorband-50": "PriorBand(50%)",
    "priorband_bo": "PriorBand+BO",
    "priorband_bo-50": "PriorBand(50%)+BO",
    "priorband_bo_linear": "PriorBand(HS+dyna)",
    "priorband_bo_cr_dyna": "PriorBand-Cr-dyna",
    "priorband_bo_hs_decay": "PriorBand-HS-decay",
    "priorband_crossover_constant": "PB(Cr+const.)",
    "priorband_crossover_constant_linear": "PriorBand(Const.)",
    "priorband_crossover_decay": "PB(Cr+decay)",
    "priorband_crossover_decay_linear": "PriorBand(Decay)",
    "priorband_crossover_dynamic": "PB-Cr+Dyn+G",
    "priorband_crossover_dynamic-default-at-target": "PriorBand(crossover)",
    "priorband_crossover_dynamic_linear": "PB(Cr+dyna+linear)",
    "priorband_hypersphere_constant": "PB(HS+const.)",
    "priorband_hypersphere_constant_linear": "PB(HS+const)",
    "priorband_hypersphere_decay": "PB(HS+decay)",
    "priorband_hypersphere_decay_linear": "PB(HS+decay)",
    "priorband_hypersphere_dynamic": "PB(HS+dyna)",
    "priorband_hypersphere_dynamic-default-at-target": "PriorBand(hypersphere)",
    "priorband_hypersphere_dynamic_linear": "PriorBand(HS)",
    "priorband_noinc_prior": "HB+Prior(Geom)",  # "PB-inc(+Pr)",
    "priorband_noinc_prior_linear": "HB+Prior(linear)",
    "priorband_noinc_random": "PB-inc(+RS)",
    "priorband_noprior_inc_cr": "HB+inc(Cr)",
    "priorband_noprior_inc_cr_linear": "HB+inc(Cr)",
    "priorband_noprior_inc_hs": "HB+inc(HS)",
    "priorband_noprior_inc_hs_linear": "HB+inc(HS)",
    # post-mutation
    "hyperband_prior_geom": "PriorBand(w/o inc)",
    "pb_hypersphere_dynamic_50": "PB-HS+Dyn+50",
    "pb_hypersphere_dynamic_geometric": "PB-HS+Dyn+G",
    "pb_hypersphere_dynamic_linear": "PB-HS+Dyn+L",
    "pb_mutation_constant_50": "PB-Mu+Con+50",
    "pb_mutation_constant_geometric": "PB-Mu+Con+G",
    "pb_mutation_constant_geometric-default-at-target": "PriorBand(constant)",
    "pb_mutation_constant_linear": "PB-Mu+Con+L",
    "pb_mutation_decay_50": "PB-Mu+Dec+50",
    "pb_mutation_decay_geometric": "PB-Mu+Dec+G",
    "pb_mutation_decay_geometric-no-default": "PB-Mu+Dec+G(No-Mode)",
    "pb_mutation_decay_geometric-default-at-target": "PriorBand(decay)",
    "pb_mutation_decay_geometric_bo": "PB-Mu+Dec+G+BO",
    "pb_mutation_decay_geometric_bo-no-default": "PB-Mu+Dec+G+BO(No-Mode)",
    "pb_mutation_decay_geometric_bo-default-at-target": "PB-Mu+Dec+G+BO(mode@max)",
    "pb_mutation_decay_linear": "PB-Mu+Dec+L",
    "pb_mutation_dynamic_50": "PB-Mu+Dyn+50",
    "pb_mutation_dynamic_50-default-at-target": "PriorBand(50%)",
    "pb_mutation_dynamic_50_bo": "PB-Mu+Dyn+50+BO",
    "pb_mutation_dynamic_geometric": "PriorBand(Mode@min)",
    "pb_mutation_dynamic_geometric-no-default": "PriorBand(No-Mode)",
    "pb_mutation_dynamic_geometric-default-at-target": "PriorBand",
    "pb_mutation_dynamic_geometric_bo": "PriorBand+BO(Mode@min)",
    "pb_mutation_dynamic_geometric_bo-no-default": "PriorBand+BO(No-Mode)",
    "pb_mutation_dynamic_geometric_bo-default-at-target": "PriorBand+BO",
    "pb_mutation_dynamic_linear": "PB-Mu+Dyn+L",
    "pb_mutation_dynamic_linear-default-at-target": "PriorBand(linear)",
    "pb_mutation_dynamic_linear_bo": "PB-Mu+Dyn+L+BO",
    "asha_pb_mut_geom_dyna": "ASHA-PB-Mu+Dyn+G",
    "asha_pb_mut_geom_dyna-default-at-target": "ASHA(+E)",
    "asha_pb_mut_geom_decay": "ASHA-PB-Mu+Dec+G",
    "asha_hb_pb_mut_geom_dyna": "Async-HB-PB-Mu+Dyn+G",
    "asha_hb_pb_mut_geom_dyna-default-at-target": "Async-HB(+E)",
    "asha_hb_pb_mut_geom_decay": "Async-HB-PB-Mu+Dec+G",
    "asha_hb_pb_mut_geom_dyna_bo_rung": "Mobster-PB-Mu+Dyn+G",
    "asha_hb_pb_mut_geom_decay_bo_rung": "Mobster-PB-Mu+Dec+G",
    "asha_hb_pb_mut_geom_dyna_bo_rung-default-at-target": "Mobster(+E)",
    "asha_hb_pb_mut_geom_dyna_bo_joint": "Mobster-PB-Mu+Dyn+G(j)",
    "asha_hb_pb_mut_geom_dyna_bo_joint-default-at-target": "Mobster(+E)[joint]",
    "asha_hb_pb_mut_geom_decay_bo_joint": "Mobster-PB-Mu+Dec+G(j)",
}


# https://matplotlib.org/stable/gallery/color/named_colors.html
COLOR_MARKER_DICT = {
    "random_search": "palevioletred",
    "random_search_prior": "magenta",
    "random_search_prior-no-default": "magenta",
    "bo-10": "darkgoldenrod",
    "pibo-no-default": "dodgerblue",
    "pibo-default-first-10": "firebrick",
    "bohb": "darkcyan",
    "mobster": "teal",
    "hyperband": "limegreen",
    "hyperband_prior": "darkgreen",
    "hyperband_prior-50": "olive",
    "hyperband_prior-no-default": "darkgreen",
    "hyperband_prior-default-at-target": "olive",
    "hb_inc": "orange",
    "hb_inc-50": "orange",
    "asha": "goldenrod",
    "asha_prior": "firebrick",
    "asha_prior-default-at-target": "firebrick",
    "asha_priorband": "seagreen",
    "asha_pb_mut_geom_dyna-default-at-target": "seagreen",
    "asha_hyperband": "peru",
    "asha_hyperband_prior": "darkolivegreen",
    "asha_hyperband_prior-default-at-target": "darkolivegreen",
    "asha_hb_pb_mut_geom_dyna-default-at-target": "indigo",
    "asha_hb_priorband": "indigo",
    "asha_hb_priorband_bo_rung": "tomato",
    "asha_hb_pb_mut_geom_dyna_bo_rung-default-at-target": "tomato",
    "asha_hb_priorband_bo_joint": "greenyellow",
    "asha_hb_pb_mut_geom_dyna_bo_joint-default-at-target": "greenyellow",
    # PriorBand
    "priorband": "blue",
    "priorband-50": "darkseagreen",
    "priorband_bo": "purple",
    "priorband_bo-50": "firebrick",
    "priorband_crossover_constant_linear": "darkkhaki",
    "priorband_crossover_decay": "orangered",
    "priorband_crossover_decay_linear": "orangered",
    "priorband_crossover_dynamic": "deeppink",
    "priorband_crossover_dynamic-default-at-target": "deeppink",
    "priorband_crossover_dynamic_linear": "deeppink",
    "priorband_hypersphere_decay": "crimson",
    "priorband_hypersphere_decay_linear": "crimson",
    "priorband_hypersphere_dynamic": "crimson",
    "priorband_hypersphere_dynamic-default-at-target": "crimson",
    "priorband_hypersphere_dynamic_linear": "crimson",
    # post-mutation
    "hyperband_prior_geom": "seagreen",
    "pb_hypersphere_dynamic_50": "darkkhaki",
    "pb_hypersphere_dynamic_geometric": "darkkhaki",
    "pb_hypersphere_dynamic_linear": "darkkhaki",
    "pb_mutation_constant_50": "crimson",
    "pb_mutation_constant_geometric": "crimson",
    "pb_mutation_constant_geometric-default-at-target": "crimson",
    "pb_mutation_constant_linear": "crimson",
    "pb_mutation_decay_50": "crimson",
    "pb_mutation_decay_geometric": "purple",
    "pb_mutation_decay_geometric-no-default": "pink",
    "pb_mutation_decay_geometric-default-at-target": "mediumturquoise",
    "pb_mutation_decay_geometric_bo": "purple",
    "pb_mutation_decay_geometric_bo-no-default": "pink",
    "pb_mutation_decay_geometric_bo-default-at-target": "mediumturquoise",
    "pb_mutation_decay_linear": "darkkhaki",
    "pb_mutation_dynamic_50": "darkseagreen",
    "pb_mutation_dynamic_50-default-at-target": "darkseagreen",
    "pb_mutation_dynamic_50_bo": "darkseagreen",
    "pb_mutation_dynamic_geometric": "peru",
    "pb_mutation_dynamic_geometric-no-default": "pink",
    "pb_mutation_dynamic_geometric-default-at-target": "blue",
    "pb_mutation_dynamic_geometric_bo": "peru",
    "pb_mutation_dynamic_geometric_bo-no-default": "pink",
    "pb_mutation_dynamic_geometric_bo-default-at-target": "purple",
    "pb_mutation_dynamic_linear": "orangered",
    "pb_mutation_dynamic_linear-default-at-target": "orangered",
    "pb_mutation_dynamic_linear_bo": "orangered",
    "asha_pb_mut_geom_dyna": "seagreen",
    "asha_pb_mut_geom_decay": "seagreen",
    "asha_hb_pb_mut_geom_dyna": "indigo",
    "asha_hb_pb_mut_geom_decay": "indigo",
    "asha_hb_pb_mut_geom_dyna_bo_rung": "tomato",
    "asha_hb_pb_mut_geom_decay_bo_rung": "tomato",
    "asha_hb_pb_mut_geom_dyna_bo_joint": "greenyellow",
    "asha_hb_pb_mut_geom_decay_bo_joint": "greenyellow",
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

DEFAULT_LINE_STYLE = {
    "color": "black",
    "linestyle": ":",
    "linewidth": 1.0,
    "dashes": (5, 10),
}

CUSTOM_MARKERS = {
    "random_search": "o",
    "hyperband": "^",
    "hyperband_prior": "v",
    "hyperband_prior-50": ".",
    "priorband": "*"
}

RC_PARAMS = {
    "text.usetex": False,  # True,
    # "pgf.texsystem": "pdflatex",
    # "pgf.rcfonts": False,
    # "font.family": "serif",
    # "font.serif": [],
    # "font.sans-serif": [],
    # "font.monospace": [],
    "font.size": "10.90",
    "legend.fontsize": "9.90",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.title_fontsize": "small",
    # "bottomlabel.weight": "normal",
    # "toplabel.weight": "normal",
    # "leftlabel.weight": "normal",
    # "tick.labelweight": "normal",
    # "title.weight": "normal",
    # "pgf.preamble": r"""
    #    \usepackage[T1]{fontenc}
    #    \usepackage[utf8x]{inputenc}
    #    \usepackage{microtype}
    # """,
}
