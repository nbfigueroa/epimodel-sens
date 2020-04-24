### AFMC model
  * **Model type:** An extended SEIR (with disease related mortality rates) which includes a Quarantine compartment that transitions from the infectedes; i.e. SEIQR. Control/predictions of lockdown policies are modeled by increasing the percentage of quarantined individuals as well as the quarantine period. The analysis considers hospital and ICU capacity.
  * **Dataset:** The data of infected people is obtained from the website [Worldometers](https://www.worldometers.info/coronavirus/) to estimate R0 and growth rate parameters. 
  * **References:** Paper found [here](https://www.sciencedirect.com/science/article/pii/S0377123720300605?via%3Dihub), code not available, but we have an implementation of it [does not match results].

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

    # Values from March 21st
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
    I0          = ((1-q)/(q)) * Q0  

    # The initial number of exposed E(0) is not defined in report, how are they computed?
    contact_rate = 10                     # number of contacts an individual has per day
    E0           = (contact_rate - 1)*I0  # Estimated exposed based on contact rate and inital infected

```

```python
# The SEIR model differential equations with mortality rates and quarentine
def seir_mumbai(t, X, N, beta, gamma, sigma, m, q, tau_q):
    S, E, I, Q, R, D = X

    # Original State equations for SEIR
    dSdt  = - (beta*S*I)/N 
    dEdt  = (beta*S*I)/N - sigma*E    

    # Incorporating Quarantine components
    dIdt  = sigma*E - gamma*I - q*I - m*I
    dQdt  = q*I - tau_q*Q - m*Q
    dRdt  = gamma*I + tau_q*Q
    dDdt  = m*I + m*Q 

    return dSdt, dEdt, dIdt, dQdt, dRdt, dDdt
```

