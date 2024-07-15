# spa24-mars-rover
Repo for holding the mars rover code.



Below are instructions for interfacing Nengo with CoppeliaSim which were created by Dr. Brent Komer.

# Using Nengo with CoppeliaSim

## Installation

Note: I have only tested installation on Ubuntu 20.04

Go to http://www.coppeliarobotics.com/downloads.html

There is a free educational version. When you click download, make sure you select the correct version for your system.

Install the remote API for Python

```
pip install coppeliasim-zmqremoteapi-client
```

You may also need these other dependencies:

```
sudo apt install xsltproc
sudo apt install libzmq3-dev
sudo apt install libboost-all-dev

pip install pyzmq
pip install cbor
pip install xmlschema
```

To launch the similator, navigate to the directory that you downloaded and run:

```
./coppeliaSim.sh
```


## Scene Files for the Tutorial

Download those [here](https://www.dropbox.com/sh/d5uhpu0inp1p4jo/AABFN2Eo3cIHfF6F5I3p3_Pza?dl=0). You can download each of them individually, or all at once by multi-selecting them. The one we will be using in this tutorial is `ss_pioneer.ttt`. The associated Nengo scripts for the other scenes have not yet been converted to the new API format that CoppeliaSim uses, but you can still use them with the legacy V-REP API or convert them yourself if you are interested.

