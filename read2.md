
# Compute WER python program

## Steps to run

prepare pred.txt and ref.txt with transcripts.
number of lines in both files must match.
normalisation of text is done using whisper's english text normalisation steps
generate the WER score by running the following commands:

```python -m venv venv```

```source venv/bin/activate```

```pip install -r requirements.txt```

```python main.py pred.txt ref.txt True```

```deactivate```
