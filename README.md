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
(venv) python manage.py [experiment name]
```

e.g.
```bash
(venv) python manage.py ClipperVSRanger
```

# Overall Documentation

This code is done using abstract factory pattern. Each Experiment has some hooks to create models, 
fault injection configurations, etc. To add another experiment you have to extend `ExperimentBase` and implement
the desired behaviour for it. 

A good example of implementing this class can be found as `ClipperVSRanger` experiment.
