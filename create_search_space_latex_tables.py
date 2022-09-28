from dataclasses import dataclass
from textwrap import indent as _indent

import mfpbench
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    Hyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)


def indent(s) -> str:
    return _indent(s, " " * 4)


def wrap(o, s="{") -> str:
    if s == "{":
        return "{" + str(o) + "}"
    raise NotImplementedError()


def esc(s) -> str:
    return s.replace("_", r"\_")


def mbox(s) -> str:
    return s.replace("-", r"\mbox{-}")


@dataclass
class Row:
    hp: Hyperparameter

    @property
    def name(self) -> str:
        return self.hp.name

    @property
    def type(self) -> str:
        hp = self.hp
        mapping = {
            FloatHyperparameter: "continuous",
            IntegerHyperparameter: "integer",
            CategoricalHyperparameter: "categorical",
            OrdinalHyperparameter: "ordinal",
            Constant: "constant",
        }
        for cls, v in mapping.items():
            if isinstance(hp, cls):
                return v

        raise NotImplementedError(self.hp)

    @property
    def values(self) -> str:
        hp = self.hp
        if isinstance(hp, (FloatHyperparameter, IntegerHyperparameter)):
            start = r"-\infty" if hp.lower is None else str(hp.lower)
            end = r"\infty" if hp.upper is None else (hp.upper)
            return mbox(f"$[{start}, {end}]$")
        elif isinstance(hp, CategoricalHyperparameter):
            return "{" + ",".join(map(str, hp.choices)) + "}"
        elif isinstance(hp, OrdinalHyperparameter):
            return "{" + ",".join(map(str, hp.sequence)) + "}"
        elif isinstance(hp, Constant):
            return (
                f"${hp.value}$" if isinstance(hp.value, (float, int)) else str(hp.value)
            )
        else:
            raise NotImplementedError

    @property
    def info(self) -> str:
        hp = self.hp
        if getattr(hp, "log", False):
            return "log"
        else:
            return ""


@dataclass
class Table:
    space: ConfigurationSpace
    alignment = ("l", "l", "l", "l")
    cols = ("name", "type", "values", "info")

    def __repr__(self) -> str:
        rows = [Row(hp) for hp in self.space.get_hyperparameters()]

        cols = self.cols
        values = [tuple(getattr(row, col) for col in cols) for row in rows]

        table_start = r"\begin{tabular}" + wrap(" ".join(self.alignment))
        table_end = r"\end{tabular}"

        header = " & ".join(["\\textbf" + wrap(c) for c in cols])
        row_strs = [" & ".join(vs) for vs in values]
        body = "\n".join([indent(r + r" \\") for r in row_strs])

        table = "\n".join(
            [
                table_start,
                r"\toprule",
                header + r" \\",
                r"\midrule",
                body,
                r"\bottomrule",
                table_end,
            ]
        )
        return esc(table)


if __name__ == "__main__":
    for kwargs in [
        dict(name="lcbench", task_id="34539"),
        dict(name="mfh3"),
        dict(name="mfh6"),
        dict(name="jahs_cifar10"),
        dict(name="lm1b_transformer_2048"),
        dict(name="uniref50_transformer_128"),
        dict(name="translatewmt_xformer_64"),
    ]:
        b = mfpbench.get(**kwargs)
        table = Table(b.space)
        print(kwargs['name'])
        print(table)
