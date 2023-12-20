## Solution based on LaBSE
In this solution pretrained LaBSE_ru_en model was used to convert textual data into numerical values (vectorization).

### Files
- `Procept_labse_small.py`: file containing the main function for predicting n most similar customer product names   
- `requirements.txt`: file with a list of dependencies for installing the necessary libraries
  
## Usage
To install the necessary libraries, run:
```sh
$ pip install -r requirements.txt
```
Model:
```sh
$ Procept_labse_ru_en.py
```
The file Procept_labse_ru_en.py describes a function that takes two lists of dictionary parameters (customer database and parsed data from dealer platforms) and returns a dictionary with recommendations.
