### Setup
If you have the prerequisites (packages in requirements.txt) installed, Run `unit-tests.py` to make sure the experiment's dependencies are ready, You should get something like:
```
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
4.0.1
1.20.1
1.0.1
2.3.0
3.3.4
1.6.2
2.10.0
```  

-----

If not, here are some options to prepare the environment for the experiment..

#### Option 1. (conda + new env = recommended)
1. Install [Anaconda](https://www.anaconda.com/) (or miniconda), to be able to use *conda* package manager.
2. Start -> Run **Anaconda Prompt** as admin
3. Run the following commands one by one..
    ```shell script
    conda create -n auto_pilot_env python=3.8 tensorflow==2.5.0
    conda activate auto_pilot_env
    pip install opencv-python
    conda install -c conda-forge matplotlib==3.3.4
    conda install -c conda-forge scikit-learn==1.0
   ```
    
4. Finally, you can check that numpy, scipy, h5py are installed via
    ```shell script
    conda list numpy
    conda list scipy
    conda list h5py
    ```

#### Option 2. (python3.8 + pip)
1. Install python 3.8 from the [download page](https://www.python.org/downloads/release/python-380/).
2. use `pip3 install <package-name>` to install the required packages, For example: `pip3 install tensorflow==2.5.0`

### Credits
This project take benefit from the following projects 
1. [Autopilot](https://github.com/akshaybahadur21/Autopilot) 
2. [Foggy-Cycle-GAN](https://github.com/ghaiszaher/Foggy-CycleGAN)