## EpiModel-Sens: Sensitivity Analysis of Compartmental Epidemic Models
Code used to evaluate standard compartmental epidemic models (SIR/SEIR) as well as novel extensions to such models used for the COVID-19 pandemic ([Cambridge](https://github.com/rajeshrinet/pyross) & [Michigan](https://github.com/lilywang1988/eSIR) model). The main problem with existing models is that they all provide a varied range of predictions which make it hard for policy-makers to decide courses of action to control the epidemic. To make sense of this, in this repo we reproduce these models and apply sensitivity analysis techniques to understand what drives differences across models and how sensitive they are to parameter and data uncertainty.

---

### Compartmental Epidemic Models
**SIR (Susceptible-Infected-Recovered) type**
  * *vanilla SIR model*: The standard SIR model.

  * *extended SIR (eSIR) model*: A standard SIR model extended with a time-varying beta (transmission rate) which can be used to assess the impact of social distancing (non-pharma interventions) controls. This is implemented by scaling a nominal beta value with a time varying function pi(t), hence, beta(t) = beta_0pi(t)  

  * *Michigan eSIR model*: An eSIR model that is formulated as stochastic state-space model. Offers two types of models, one with quarantine compartment (stoch-eSQIR) and one without (stoch-eSIR).
    * **Details:** Estimates r0, infection period from observed data so far, accepts prior distributions over these parameters to represent known values, uses time varying quarantine and growth rate parameters. Does not justify why particular growth rate scaling is used for particular intervention strategies. [paper]( https://doi.org/10.1101/2020.02.29.20029421) and [code](https://github.com/lilywang1988/eSIR).

* *Cambridge SIR model*: An age and social-contact structured SIR model. It can assess the impact of social distancing (non-pharma interventions) controls by modifying the contact structures.
    * **Details:** The data of infected people is obtained from the website [Worldometers](https://www.worldometers.info/coronavirus/), [paper](https://arxiv.org/pdf/2003.12055.pdf) and [code](https://github.com/rajeshrinet/pyross).

**SEIR (Susceptible-Exposed-Infected-Recovered) type**
  * *vanilla SEIR model*:

  * *extended SEIR (eSEIR) model*: Standard SEIR model with time-varying beta(t) as described for eSIR.
  
  * *AFMC SEIQR model*: An extended SEIR (with disease related mortality rates) which includes a Quarantine compartment that transitions from the infectedes; i.e. SEIQR. Control/predictions of lockdown policies are modeled by increasing the percentage of quarantined individuals as well as the quarantine period.
    * **Details:** The data of infected people is obtained from the website [Worldometers](https://www.worldometers.info/coronavirus/) to estimate R0 and growth rate parameters. Paper found [here](https://www.sciencedirect.com/science/article/pii/S0377123720300605?via%3Dihub), code not available, but we have an implementation of it [does not match results].

---

### Parameter Estimation and Extrapolation Techniques
* Luis. B code for extrapolation, uses data from [here](https://hgis.uw.edu/virus/) csv file can be updated from this [link](https://github.com/jakobzhao/virus/blob/master/assets/virus.csv)

## Installation


## Usage


## Contact
Contributors: Nadia Figueroa and Ankit Shah.  
Advisors: David Kaiser and Julie Shah.

## References




