# MIT-COVID19-Model-Evaluation

Repository for code to evaluate the COVID-19 Models being used by different institutes and proposed to the Indian government. 

Models provided to Indian Government and info on what is needed to reproduce each model: 
* CDDEP (aka Johns Hopkins) model (SIR/SEIR model with  ?)
  * Don't know model structure yet
  * Presumably uses CCSE Johns Hopkins data which we have.
  
* Cambridge model (Age-structured SIR model): Includes age and social contact structure of the Indian population when assessing the impact of social distancing [arxiv-link](https://arxiv.org/pdf/2003.12055.pdf), predicts 21-day lockdown is not enough to prevent resurgence in virus spread.  
  * Model structure defined in paper, seems easy to implement (SIR + age-structured transition dynamics)
  * The data of infected people is obtained from the website [Worldometers](https://www.worldometers.info/coronavirus/), we currently only have the Johns Hopkins Pull.
  * Population data and contact surveys to construct Contact matrices (references in paper).
  
* UMichigan model (SIR/SEIR model with + ?)
  * Don't know model structure yet
  * Don't know which data they use

These models provide a varied range of predictions which make it hard for policy-makers to decide courses of action to control the epidemic. In this repo we will reverse-engineer the models and apply sensitivity analysis techniques to understand what drives such drastic differences across models.


Other known models being used by governments/media:
 * Imperial College model (Stochastic agent/Individual-based model) ??
 * Oxford study (SEIR with Bayesian estimates) ??




