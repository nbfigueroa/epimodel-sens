# MIT-COVID19-Model-Evaluation

Repository for code to evaluate the COVID-19 Models being used by different institutes and proposed to the Indian government. The main problem with existing models is that they all provide a varied range of predictions which make it hard for policy-makers to decide courses of action to control the epidemic. In this repo we will reverse-engineer the models and apply sensitivity analysis techniques to understand what drives such drastic differences across models.

### Modelling Techniques
Models provided to Indian Government and info on what is needed to reproduce each model:   
* Cambridge model (Age and Contact structured SIR model): Includes age and social contact structure of the Indian population when assessing the impact of social distancing, predicts 21-day lockdown is not enough to prevent resurgence in virus spread.  
  * Model structure defined in paper [arxiv-link](https://arxiv.org/pdf/2003.12055.pdf), seems easy to implement (SIR + age-structured transition dynamics)
  * The data of infected people is obtained from the website [Worldometers](https://www.worldometers.info/coronavirus/), we currently only have CCSE.
  * Code, Population data and contact surveys to construct Contact matrices available [online]().
  
* Mumbai/Pune model (SEIR + Quarantine compartment): 
  * Model stucture defined in paper [paper](https://www.sciencedirect.com/science/article/pii/S0377123720300605?via%3Dihub)
  
* UMichigan model (extended SIR model; i.e. state-space)
  * Don't know model structure yet
  * CCSE Johns Hopkins data which we have access to
  * Estimates r0, infection period from observed data so far, accepts prior distributions over these parameters to represent known values, uses time varying quarantine and growth rate parameters. 
  * Does not justify why particular growth rate scaling is used for particular intervention strategies
  * Code implemented in R, 

* CDDEP (aka Johns Hopkins) model (will not evaluate now due to lack of transperency)
  * Don't know model structure yet
  * Presumably uses CCSE Johns Hopkins data which we have access to.

Other known models being used by other governments/media:
 * Imperial College model (Stochastic agent/Individual-based model) ??
 * Oxford study (SEIR with Bayesian estimates) ??

### Estimation and Extrapolation Techniques
* Luis. B code for extrapolation, uses data from [here](https://hgis.uw.edu/virus/) csv file can be updated from this [link](https://github.com/jakobzhao/virus/blob/master/assets/virus.csv)

### Data Sources

