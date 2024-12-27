import re
import typer
from jiwer import process_words
from whisper_norm import EnglishTextNormalizer
from typing import Dict, List, Optional
from html_generator import HTMLGenerator

app = typer.Typer()
normalizer = EnglishTextNormalizer()

class WERCalculator:
    def __init__(self, html_output: bool = True):
        """Initialize WER calculator with HTML output option."""
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

    def compute_wer(self, hypothesis: str, reference: str) -> Dict:
        """Compute WER and generate alignment visualization using jiwer."""
        # Process words and get alignment information
        result = process_words(reference, hypothesis)
        
        # Reset error tracking for this comparison
        aligned_html = []
        
        # Get the individual words
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Process each alignment chunk in original order
        for chunk in result.alignments[0]:  # Remove the reverse here
            if not hasattr(chunk, 'type'):
                continue
                
            if chunk.type == 'equal':
                # Words match
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    word = hyp_words[i]
                    aligned_html.append(self._generate_html(word, word, "none"))
                    self.wer_info['nw'] += 1
                    
            elif chunk.type == 'delete':
                # Words were deleted from reference
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    word = ref_words[i]
                    aligned_html.append(self._generate_html("", word, "del"))
                    self.error_examples['del'].append(word)
                    self.wer_info['del'] += 1
                    self.wer_info['nw'] += 1
                    
            elif chunk.type == 'insert':
                # Words were inserted in hypothesis
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    word = hyp_words[i]
                    aligned_html.append(self._generate_html(word, "", "ins"))
                    self.error_examples['ins'].append(word)
                    self.wer_info['ins'] += 1
                    
            elif chunk.type == 'substitute':
                # Words were substituted
                for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx),
                            range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                    ref_word = ref_words[i]
                    hyp_word = hyp_words[j]
                    aligned_html.append(self._generate_html(hyp_word, ref_word, "sub"))
                    self.error_examples['sub'].append((hyp_word, ref_word))
                    self.wer_info['sub'] += 1
                    self.wer_info['nw'] += 1
        
        if self.html_output:
            # Don't reverse the HTML - just join it directly
            self.aligned_htmls.append(''.join(aligned_html))
        
        return self._calculate_stats()

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

def read_file(file_path: str, normalize: bool = False, remove_comments: bool = False) -> List[str]:
    """Read and process file contents using Whisper normalization."""
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file]
    
    processed_lines = []
    for text in lines:
        if remove_comments:
            # Remove comments surrounded by box brackets before normalization
            text = re.sub(r'\[\w+\]', '', text)
        if normalize:
            text = normalizer(text)
        processed_lines.append(text)
    
    return processed_lines

@app.command()
def main(
    prediction_file: str,
    reference_file: str,
    normalize: bool = typer.Option(True, "--normalize", "-n", help="Apply Whisper text normalization"),
    remove_comments: bool = typer.Option(True, "--remove-comments", "-r", help="Remove comments in brackets"),
    html_output: bool = typer.Option(True, "--html", help="Generate HTML visualization")
):
    """Calculate Word Error Rate (WER) between prediction and reference files."""
    # Read and normalize files
    predictions = read_file(prediction_file, normalize, remove_comments)
    references = read_file(reference_file, normalize, remove_comments)
    
    print("\n" + "=" * 50)
    print(f"Configuration: normalize={normalize}, remove_comments={remove_comments}")
    
    # Initialize calculator
    calculator = WERCalculator(html_output=html_output)
    calculator.reset_stats()
    
    # Process all segments
    stats = None
    for i, (hyp, ref) in enumerate(zip(predictions, references), 1):
        print(f"\nProcessing segment {i}:")
        print(f"Hypothesis: {hyp[:50]}...")
        print(f"Reference:  {ref[:50]}...")
        stats = calculator.compute_wer(hyp, ref)
    
    if stats:
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
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    app()