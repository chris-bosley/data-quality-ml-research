========================
Data Quality ML Research
========================






Project files and data to support research on impact of degrading data quality on pretrained ML models.

Configuration Specification
--------
Runs are to be configured with a json file specified when calling the package from the command line.

Example configuration
________

```
{
"data_location": "../data/online_shoppers_intention.csv",
"model": "Logistic Regression",
"mutator": "add_noise",
"num_mutation_checks": 10,
"mutation_target": "ExitRates",
"num_scoring_groups": 5,
"seeds": {
"outer_seed": 123,
"inner_seed": 456
},
"training_percent": 0.5,
"noise_arguments": {"distribution": "exponential", "mean": 0, "std": 1, "low": 0, "high": 100, "scale":2}
}
```

* Fields
    * data_location (str)
        * Specifies path to data file used to train & score model
        * Ex: "../data/online_shoppers_intention.csv"
    * model (str)
        * Specifies model type to be used
        * Valid models: Logistic Regression, Poly SVC, Linear SVC, RBF SVC, Random Forest, Neural Network
    * mutator (str)
        * Specifies mutator to apply
        * Valid mutators: change_to_zero, transpose_columns, add_categorical_class, remove_class, add_noise
    * num_mutation_checks (int)
        * Specifies number of mutation checks to run. This will also determine the intervals used (i.e., 10 mutation checks will create 10 bins beginning with 0.00 and ending at 0.9 every 0.1)
    * mutation_target (str or iterable)
        * Targets for the mutation. Mutation itself determines whether a string or iterable is accepted
            * change_to_zero: Accepts string or iterable
            * transpose_columns: Accepts iterable, length 2
            * add_categorical_class: Accepts string
            * remove_class: Accepts string
            * add_noise: accepts string
    * num_scoring_groups (int)
        * Specifies the number of groups to create for mutation checks. Will evenly divide all non-training data.
    * seeds (map<str, int>)
        * outer_seed (int)
        * inner_seed (int)
        * Should do something here but... it's a mess. Just leave the names haha.
    * training_percent (float)
        * Must be in (0,1)
        * Specifies proportion of dataset used for training
    * noise_arguments (map<str, int/float/string<) (only required for add_noise)
        * This is a mess but it works for now
        * distribution (string)
            * Allowed values: normal, exponential, uniform
        * mean (numeric)
            * mean for normal distribution noise
        * std (numeric)
            *  standard deviation for normal distribution noise
        * low (numeric)
            * lower bound for uniform noise
        * high (numeric)
            * upper bound for uniform noise
        * scale (numeric)
            * scale for exponential noise

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
