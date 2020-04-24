## EpiSens: Sensitivity Analysis of Compartmental Epidemic Models
Code used to evaluate standard compartmental epidemic models (SIR/SEIR) as well as novel extensions to such models used for the COVID-19 pandemic (Cambridge & Michigan model). The main problem with existing models is that they all provide a varied range of predictions which make it hard for policy-makers to decide courses of action to control the epidemic. 

To make sense of this, in this repo we implement standard models, simple extensions and more complex extensions (further compartments and stochastic estimates) and apply sensitivity analysis techniques to understand what drives differences across models and how sensitive they are to parameter estimates.

### Models
* SIR model: 

* SEIR model:

* extended SIR (eSIR) model: time-varying beta (to simulate social distancing protocols/measures)

* extended SEIR (eSEIR) model: time-varying beta (to simulate social distancing protocols/measures)

* Cambridge model: 
  * **Model type:** An extended SIR model that includes age and social contact structure. It can assess the impact of social distancing, predicts 21-day lockdown is not enough to prevent resurgence in virus spread. The model has been extended to SEIR form.
  * **Dataset:** The data of infected people is obtained from the website [Worldometers](https://www.worldometers.info/coronavirus/)
  * **References:** Paper found in [arxiv-link](https://arxiv.org/pdf/2003.12055.pdf), code implemented in Python found in [github-link](https://github.com/rajeshrinet/pyross).

* UMichigan model 
  * **Model type:** The "Michigan Model" is an extended SIR that is formulated as stochastic state-space model. 
  * CCSE Johns Hopkins data which we have access to
  * Estimates r0, infection period from observed data so far, accepts prior distributions over these parameters to represent known values, uses time varying quarantine and growth rate parameters. 
  * Does not justify why particular growth rate scaling is used for particular intervention strategies
  * Code implemented in R, 

* AFMC model:
  * **Model type:** An extended SEIR (with disease related mortality rates) which includes a Quarantine compartment that transitions from the infectedes; i.e. SEIQR. Control/predictions of lockdown policies are modeled by increasing the percentage of quarantined individuals as well as the quarantine period. The analysis considers hospital and ICU capacity.
  * **Dataset:** The data of infected people is obtained from the website [Worldometers](https://www.worldometers.info/coronavirus/) to estimate R0 and growth rate parameters. 
  * **References:** Paper found [here](https://www.sciencedirect.com/science/article/pii/S0377123720300605?via%3Dihub), code not available, but we have an implementation of it [does not match results].


### Parameter Estimation and Extrapolation Techniques
* Luis. B code for extrapolation, uses data from [here](https://hgis.uw.edu/virus/) csv file can be updated from this [link](https://github.com/jakobzhao/virus/blob/master/assets/virus.csv)

## Installation


## Usage


## Contact
Contributors: Nadia Figueroa and Ankit Shah.
Advisors: David Kaiser and Julie Shah.

## Publications




