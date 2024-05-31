# pipeline
contains the pipeline to process the raw data 

# How to use

## install python 
download and install python from https://www.python.org/downloads/

## create a virtual environment 
within the pipeline folder run ```python -m venv .venv``` (on windows)

## activate the virtual environment
| Platform      | Shell | Command to activate virtual environment |
| ----------- | ----------- | ----------- |
| POSIX     | bash/zsh      | ```$ source <venv>/bin/activate```        |
|           | fish          | ```$ source <venv>/bin/activate.fish```   |
|           | csh/tcsh      | ```$ source <venv>/bin/activate.csh```    |
|           | PowerShell    | ```$ <venv>/bin/Activate.ps1```           |
| Windows   | cmd.exe       | ```C:\> <venv>\Scripts\activate.bat```    |
|           | PowerShell    | ```PS C:\> <venv>\Scripts\Activate.ps1``` |

## install requirements in the virtual environment
```python -m pip install -r requirements.txt```

## open this pipeline in vscode
within the pipeline folder run ```code .```
make sure extensions for python and jupyter support are installed 

## run the notebook 
* open data.ipynb
* select the right kernel (either on the top right of the notebook or by using STRG+SHIFT+P and using 'Notebook: Select notebook kernel')
	




	

