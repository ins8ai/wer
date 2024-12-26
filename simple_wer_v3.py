"""wer_calculator.py: Main module for WER calculation."""

import re
import typer
from evaluate import load
from whisper_norm import EnglishTextNormalizer
from typing import Dict, List, Optional
from html_generator import HTMLGenerator

app = typer.Typer()
wer_metric = load("wer")
normalizer = EnglishTextNormalizer()

class WERCalculator:
    def __init__(self, html_output: bool = True):
        self.html_output = html_output
        self.aligned_htmls: List[str] = []
        self.error_examples = {'sub': [], 'ins': [], 'del': []}
        self.wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': 0}
        self.html_generator = HTMLGenerator() if html_output else None

    def reset_stats(self):
        """Reset all statistics counters."""
        self.wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': 0}
        self.error_examples = {'sub': [], 'ins': [], 'del': []}
        self.aligned_htmls = []

    def compute_wer(self, hypothesis: str, reference: str) -> Dict:
        """Compute WER and generate alignment visualization."""
        hyp_words = hypothesis.split()
        ref_words = reference.split()
        
        # Compute edit distance
        distmat = compute_edit_distance_matrix(hyp_words, ref_words)
        
        # Back trace to distinguish different errors
        pos_hyp, pos_ref = len(hyp_words), len(ref_words)
        aligned_html = []
        
        while pos_hyp > 0 or pos_ref > 0:
            hyp_word = ' ' if pos_hyp == 0 else hyp_words[pos_hyp - 1]
            ref_word = ' ' if pos_ref == 0 else ref_words[pos_ref - 1]
            
            # Determine error type
            if pos_ref == 0:
                err_type = 'ins'
                self.error_examples['ins'].append(hyp_word)
            elif pos_hyp == 0:
                err_type = 'del'
                self.error_examples['del'].append(ref_word)
            else:
                if hyp_words[pos_hyp - 1] == ref_words[pos_ref - 1]:
                    err_type = 'none'
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp - 1] + 1:
                    err_type = 'sub'
                    self.error_examples['sub'].append((hyp_word, ref_word))
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp] + 1:
                    err_type = 'del'
                    self.error_examples['del'].append(ref_word)
                else:
                    err_type = 'ins'
                    self.error_examples['ins'].append(hyp_word)
            
            if self.html_output:
                aligned_html.append(self._generate_html(hyp_word, ref_word, err_type))
            
            # Update statistics
            if err_type != 'none':
                self.wer_info[err_type] += 1
            if err_type != 'ins':
                self.wer_info['nw'] += 1
            
            # Update positions
            if err_type == 'del':
                pos_ref -= 1
            elif err_type == 'ins':
                pos_hyp -= 1
            else:
                pos_hyp -= 1
                pos_ref -= 1
        
        if self.html_output:
            self.aligned_htmls.append(''.join(reversed(aligned_html)))
        
        return self._calculate_stats()

    def _generate_html(self, hyp: str, ref: str, err_type: str) -> str:
        """Generate HTML for word alignment visualization."""
        if err_type == 'none':
            return f'{hyp} '
        
        styles = {
            'sub': f'''<span style="background-color: yellow"><del>{hyp}</del></span>
                      <span style="background-color: yellow">{ref}</span> ''',
            'del': f'''<span style="background-color: red">{ref}</span> ''',
            'ins': f'''<span style="background-color: green"><del>{hyp}</del></span> '''
        }
        return styles.get(err_type, '')

    def _calculate_stats(self) -> Dict:
        """Calculate WER statistics."""
        total_words = max(self.wer_info['nw'], 1)
        total_errors = sum(self.wer_info[k] for k in ['sub', 'del', 'ins'])
        wer = (total_errors * 100.0) / total_words
        
        return {
            'wer': wer,
            'accuracy': 100 - wer,
            'substitutions': (self.wer_info['sub'] * 100.0) / total_words,
            'deletions': (self.wer_info['del'] * 100.0) / total_words,
            'insertions': (self.wer_info['ins'] * 100.0) / total_words,
            'total_words': total_words,
            'total_errors': total_errors
        }

    def save_html(self, filename: str) -> None:
        """Save HTML visualization using the HTML generator."""
        if not self.html_output or not self.html_generator:
            return
            
        stats = self._calculate_stats()
        error_counts = {
            'sub': len(self.error_examples['sub']),
            'del': len(self.error_examples['del']),
            'ins': len(self.error_examples['ins'])
        }
        
        self.html_generator.save_html(
            filename=filename,
            stats=stats,
            aligned_htmls=self.aligned_htmls,
            error_examples=self.error_examples,
            error_counts=error_counts
        )

def compute_edit_distance_matrix(hyp_words: List[str], ref_words: List[str]) -> List[List[int]]:
    """Compute edit distance matrix using dynamic programming."""
    rows, cols = len(ref_words) + 1, len(hyp_words) + 1
    matrix = [[j if i == 0 else i if j == 0 else 0 
              for j in range(cols)] for i in range(rows)]
    
    for i in range(1, rows):
        for j in range(1, cols):
            if ref_words[i - 1] == hyp_words[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(matrix[i - 1][j - 1],  # substitution
                                 matrix[i][j - 1],        # insertion
                                 matrix[i - 1][j]) + 1    # deletion
    return matrix

def preprocess_text(txt: str, remove_comments: bool = False) -> str:
    """Preprocess text before WER calculation."""
    if remove_comments:
        txt = re.sub(r'\[\w+\]', '', txt) # Remove comments surrounded by box brackets, e.g., [comment]
    
    preprocessing_steps = [
        lambda x: x.lower(), # Convert text to lowercase for case-insensitive comparison
        lambda x: x.replace('$', ''), # Remove dollar signs from the text
        lambda x: ' '.join(word[:-1] if word.endswith('s') else word for word in x.split()), # Remove trailing 's' from words (handle plurals)
        lambda x: re.sub(r'[\t\n]', ' ', x), # Replace tabs and newlines with spaces for consistent formatting
        lambda x: re.sub(r'["()\[\]]', '', x), # Remove quotes, brackets, and parentheses
        lambda x: re.sub(r'[,.?!]+ ', ' ', x), # Remove punctuation marks before spaces
        lambda x: re.sub(r'[,.?!]+$', ' ', x), # Remove punctuation marks at the end of text
        lambda x: re.sub(r' [,.?!]+', ' ', x), # Remove punctuation marks after spaces
        lambda x: re.sub(' +', ' ', x.strip()), # Remove extra spaces and trim
    ]
    
    for step in preprocessing_steps:
        txt = step(txt)
    
    return txt

def read_file(file_path: str, normalize: bool = False, remove_comments: bool = False) -> List[str]:
    """Read and process file contents."""
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]
    
    if normalize:
        lines = [normalizer(text) for text in lines]
    if remove_comments:
        lines = [preprocess_text(text, True) for text in lines]
    
    return lines

@app.command()
def main(
    prediction_file: str,
    reference_file: str,
    normalize: bool = typer.Option(True, "--normalize", "-n", help="Apply text normalization"),
    remove_comments: bool = typer.Option(True, "--remove-comments", "-r", help="Remove comments in brackets"),
    html_output: bool = typer.Option(True, "--html", help="Generate HTML visualization"),
    huggingface_wer: bool = typer.Option(True, "--hf", help="Use HuggingFace WER calculation")
):
    """Calculate Word Error Rate (WER) between prediction and reference files."""
    predictions = read_file(prediction_file, normalize, remove_comments)
    references = read_file(reference_file, normalize, remove_comments)
    
    print("\n" + "=" * 50)
    print(f"Configuration: normalize={normalize}, remove_comments={remove_comments}")
    
    # Initialize calculator
    calculator = WERCalculator(html_output=html_output)
    calculator.reset_stats()  # Ensure clean state
    
    # Process all segments
    stats = None
    for i, (hyp, ref) in enumerate(zip(predictions, references), 1):
        print(f"\nProcessing segment {i}:")
        print(f"Hypothesis: {hyp[:50]}...")
        print(f"Reference:  {ref[:50]}...")
        stats = calculator.compute_wer(hyp, ref)
    
    if stats:  # Only print if we have valid statistics
        print(f"\nResults:")
        print(f"Word Error Rate: {stats['wer']:.3f}")
        print(f"Accuracy: {stats['accuracy']:.1f}%")
        print("\nError Breakdown:")
        print(f"Substitutions: {stats['substitutions']:.1f}%")
        print(f"Deletions: {stats['deletions']:.1f}%")
        print(f"Insertions: {stats['insertions']:.1f}%")
        print(f"\nTotal Words: {stats['total_words']}")
        print(f"Total Errors: {stats['total_errors']}")
    
        if html_output:
            calculator.save_html(prediction_file)
    
    if huggingface_wer:
        wer_score = wer_metric.compute(predictions=predictions, references=references)
        print(f"\nHuggingFace WER: {wer_score:.3f}")
        print(f"HuggingFace Accuracy: {(1.0 - wer_score) * 100:.1f}%")
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    app()