# WER

Word Error Rate (wer) is a metric used to assess and compare accuracy of transcripts generated.
WER is used to express the differences between predicted text and the reference text.
Differences include insertion, deletion and substitutions required to transform predicted text to reference text. 


This repo contains sample codes for how the ins8.ai team is evaluating WER scores.
Normalisation includes removal of punctuation, uppercase conversion to lowercase, standisation in numerical representation, translation of british to american spellings and etc. Normalisation is recommended to be enabled. You may enable it when running the script.


## Install and Pre-requisites

- after cloning the repo, create a new virtual environment and activate it
- install the required packages via pip install -r requirements.txt
- prepare pred.txt and ref.txt with transcripts and number of lines in both files must match.


## Run

- make sure your virtual environment is activated
- run the script using the following example command

``` python main.py pred.txt ref.txt True ```


