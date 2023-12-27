import pandas as pd

from include.summary import Summary


def generate_summary(custom_layers, writer, chip_arch):
    smy = Summary(writer)
    smy.load(custom_layers)
    smy.write(chip_arch)