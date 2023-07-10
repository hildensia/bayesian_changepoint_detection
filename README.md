

Bayesian Changepoint Detection
==============================

Methods to get the probability of a changepoint in a time series. Both online and offline methods are available. Read the following papers to really understand the methods:


[1] Paul Fearnhead, Exact and Efficient Bayesian Inference for Multiple                                    
    Changepoint problems, Statistics and computing 16.2 (2006), pp. 203--213                               
                                                                                                           
[2] Ryan P. Adams, David J.C. MacKay, Bayesian Online Changepoint Detection,                               
    arXiv 0710.3742 (2007)                                                                                 
                                                                                                           
[3] Xuan Xiang, Kevin Murphy, Modeling Changing Dependency Structure in                                    
    Multivariate Time Series, ICML (2007), pp. 1055--1062
    
To see it in action have a look at the [example notebook](https://github.com/hildensia/bayesian_changepoint_detection/blob/master/Example_Code.ipynb "Example Code in an IPython Notebook").


# To install:

```bash
# Enter a directory of your choice, activate your python virtual environment.
git clone https://github.com/hildensia/bayesian_changepoint_detection.git
cd bayesian_changepoint_detection
pip install .
# Now can use bayesian_changepoint_detection in python.
```

---

Or using pip - older version of this package, that doesn't work with python3:
```bash
pip install bayesian-changepoint-detection
```
