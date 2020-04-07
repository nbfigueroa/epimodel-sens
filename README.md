# MIT-COVID19-Model-Evaluation

Repository for code to evaluate the COVID-19 Models being used by different institutes and proposed to the Indian government. The main problem with existing models is that they all provide a varied range of predictions which make it hard for policy-makers to decide courses of action to control the epidemic. In this repo we will reverse-engineer the models and apply sensitivity analysis techniques to understand what drives such drastic differences across models.

### Modeling Techniques
Models provided to Indian Government and info on what is needed to reproduce each model:   
* Cambridge model: 
  * **Model type:** The "Cambridge Model" is an extended SIR model that includes and age and social contact structure of the Indian population. It can assessing the impact of social distancing, predicts 21-day lockdown is not enough to prevent resurgence in virus spread. The model has been extended to SEIR form.
  * **Dataset:** The data of infected people is obtained from the website [Worldometers](https://www.worldometers.info/coronavirus/)
  * **References:** Paper found in [arxiv-link](https://arxiv.org/pdf/2003.12055.pdf), code implemented in Python found in [github-link](https://github.com/rajeshrinet/pyross).
  
* Mumbai/Pune model
  * **Model type:** The "Mumbai/Pune Model" is an extended SEIR (with disease related mortality rates) which includes a Quarantine compartment that transitions from the infectedes; i.e. SEIQR. Control/predictions of lockdown policies is modeled by increasing the percentage of quarantined individuals as well as the quarantine period. The analysis considers hospital and ICU capacity.
  * **Dataset:** The data of infected people is obtained from the website [Worldometers](https://www.worldometers.info/coronavirus/) to estimate R0 and growth rate parameters. 
  * **References:** Paper found [here](https://www.sciencedirect.com/science/article/pii/S0377123720300605?via%3Dihub), code not available, but we have an implementation of it [does not match results].
  
* UMichigan model 
  * **Model type:** The "Michigan Model" is an extended SIR that is formulated as stochastic state-space model. 
  * Don't know model structure yet
  * CCSE Johns Hopkins data which we have access to
  * Estimates r0, infection period from observed data so far, accepts prior distributions over these parameters to represent known values, uses time varying quarantine and growth rate parameters. 
  * Does not justify why particular growth rate scaling is used for particular intervention strategies
  * Code implemented in R, 

Other known models being used by other governments/media:
 * CDDEP (aka Johns Hopkins) model (will not evaluate due to lack of transparency)
   * Don't know model structure yet
   * Presumably uses CCSE Johns Hopkins data which we have access to.
 * Imperial College model (Stochastic agent/Individual-based model)
 * Oxford study (SEIR with Bayesian estimates)

### Estimation and Extrapolation Techniques
* Luis. B code for extrapolation, uses data from [here](https://hgis.uw.edu/virus/) csv file can be updated from this [link](https://github.com/jakobzhao/virus/blob/master/assets/virus.csv)

---
Contributors: Nadia Figueroa and Ankit Shah.

