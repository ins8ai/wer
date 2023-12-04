import typer
from evaluate import load
from whisper_norm import EnglishTextNormalizer

app = typer.Typer()
wer = load("wer")
normalizer = EnglishTextNormalizer()

@app.command()
def main(prediction_file, reference_file, normalise: bool):
    predictions = read_file(prediction_file, normalise)
    references = read_file(reference_file, normalise)
    wer_score = wer.compute(predictions=predictions, references=references)
    accuracy = (1.0 - wer_score) * 100
    print("")
    print("=" * 50)
    print(f"Normalisation: {normalise}")
    print(f"Word Error Rate: {wer_score}")
    print(f"Accuracy: {accuracy:.1f}%")
    print("=" * 50)
    print("")

# Open the file in read mode and read its contents line by line
def read_file(file_path, normalise):
    lines = []
    with open(file_path, "r") as file:
        for line in file:
            if (normalise == False):
                lines.append(line.strip())
            else:
                lines.append(normalizer(line.strip()))
    return lines

if __name__ == "__main__":
    app()


