Implementation of MumbaiPune model (An SEIR model with Quarantine compartment):

Folowing are the values used to run our simulations:
```python
   
    # Initial values from March 21st for India test-case
    N            = 1375987036 # Number from report
    days         = 365
    gamma_inv    = 7  
    sigma_inv    = 5.1
    m            = 0.0043
    r0           = 2.28      
    tau_q_inv    = 14

    # Initial values from March 21st "indian armed forces predictions"
    R0           = 23
    D0           = 5         
    Q0           = 249               
    
    # This is the total number of comfirmed cases for March 21st, not used it seems?                                   
    T0           = 334               
    
    # Derived Model parameters and 
    beta       = r0 / gamma_inv
    sigma      = 1.0 / sigma_inv
    gamma      = 1.0 / gamma_inv
    tau_q      = 1.0 /tau_q_inv

    # Control variable:  percentage quarantined
    q           = 0.001
    # Q0 is 1% of total infectious; i.e. I0 + Q0 (as described in report)
    # In the report table 1, they write number of Quarantined as SO rather than Q0
    # Q0, is this a typo? 
    # Number of Infectuos as described in report    
    I0          = ((1-q)/(q)) * Q0  

    # The initial number of exposed E(0) is not defined in report, how are they computed?
    contact_rate = 10                     # number of contacts an individual has per day
    E0           = (contact_rate - 1)*I0  # Estimated exposed based on contact rate and inital infected

```
