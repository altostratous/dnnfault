# dnnfault
Some experiments on deep neural networks and fault injections

# Installation

Create whatever virtual environment and install requirements in requirements.txt.

If you're using virtualenv and pip, will go like this:

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage

```bash
(venv) python manage.py [experiment class name]
```

e.g.
```bash
(venv) python manage.py ClipperVSRanger
```

# High Level Documentation

This code is done using abstract factory pattern. Each Experiment has some hooks to create models, 
fault injection configurations, etc. To add another experiment you have to extend `ExperimentBase` and implement
the desired behaviour for it. You can add a directory to the root and register it in settings.py to add a new series of 
experiments.

A good example of implementing this class can be found as `ClipperVSRanger` experiment. You have to configure its
dataset manually by changing the `get_dataset` method. 

# Results

The default implementation for ExperimentBase logs the results in pickle format containing all samples classifications.

You can customize this behaviour by overriding the logging methods in your experiment.
