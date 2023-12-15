# helmpy - individual-based simulation of helminth transmission in a human population

An all-in-one python class for forecasting which includes features for

1. ageing of individuals between groups,
2. mass drug administration for any given coverage and drug efficacy,
3. infected human migration between separate clusters of individuals,
4. individual non-compliance to treatment,
5. fitting summary parameters to datasets,
6. and generating posterior predictions based on these datasets 

for helminth (parasitic worm) disease transmission with variable population sizes. The mathematics of the model are described in [Hardwick et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1755436521000013?via%3Dihub).

The code runs with an optimised stochastic simulation method where best results can be attained through only a minimal (typically ~10, depending on the overall population number) number of repeat runs.

## Getting started

To fork, simply type into the terminal:

> git clone https://github.com/umbralcalc/helmpy.git

In the `/notebooks` directory there is an ipython notebook with worked examples for easy implementation.
