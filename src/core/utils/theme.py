import matplotlib
import seaborn as sns


def set_theme():
    sns.set_theme(palette="deep")
    sns.set_style("whitegrid")
    sns.set_context("poster")

    # Set a font that supports a wide range of Unicode characters
    matplotlib.rcParams["font.family"] = "Arial Unicode MS"
