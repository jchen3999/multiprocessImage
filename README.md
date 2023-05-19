## A script for multiprocessing channel data.
To run the script on Jarvis (i.e., bhg0060):
1. Download the `processImages.py` file  and put it wherever you like on Bluehive (I personally recommend create a new directory for this script as it will save some cache files).

2. Login to Jarvis: 
   - within your [Bluehive interactive session](https://bluehive.circ.rochester.edu/auth/ssh), open a terminal, run `ssh bhg0060`
   - or, open a local terminal, run  `ssh yourNetID@bluehive.circ.rochester.edu`, then run `ssh bhg0060`
   
3. Setup a python environment (you only need to setup this once; and you can ignore this step if you already have a python3 environment)
    ```bash
    # create a python environment
    module load anaconda3/5.3.0b
    conda create -n some_env_name python==3.7.10
    ```
    
    ```bash
    # activate the enviroment 
    conda activate some_env_name
    # install necessary packages
    # enable a more colorful output
    pip install colorama
    # show concise gpu info
    pip install gpustat
    ```
    
4. Usage:

    - multi-processing channel data:

    - I made this script very interactive, you can follow the prompt to specify which files to process

      ```bash
      # activate the env you just set up using (you may also put this in the .bashrc file which is located at /home/yourNetID/.bashrc. so it will automatically activate this env every time you open a terminal):
      # conda activate some_env_name
      # change directory to where the `processImages.py` is located using: 
      # cd path_to_file 
      python processImages.py
      # after you entered 'python pro', you may hit `Tab` to auto-complete
      ```

    - check job progress:

      ```bash
      python processImages.py --check
      ```
    - kill running job (Note that this will terminate **ALL** Matlab process, including the ones with interfaces!):
      ```bash
      python processImages.py --kill
      ```

## About cache
Some cache files will be created at a `cache` folder under where you put this script after executing it, including argumentations used for creating the jobs, log files of the jobs, some temporary matlab/bash scripts. You can safely delete them after the jobs are finished, or leave them there as they won't take up a lot of space.


## Implemented features:

- automatically probe existing date/test/burst
- automatically exclude empty test folders/nonexistent bursts
- automatically detect potential typos (e.g., wrong exp date) and raise warning info
- interactive and colorful outputs
- run jobs in background 
- easily check progress 

## TODO:

- [x] feel free to leave any comments, like some other features to include or issues you noticed. 
