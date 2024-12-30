import re
import typer
from whisper_norm import EnglishTextNormalizer
from typing import Dict, List, Optional, Tuple
from html_generator import HTMLGenerator

app = typer.Typer()
normalizer = EnglishTextNormalizer()

class PluralNormalizer:
    """Handles normalization of plural words to their singular form."""
    
    def __init__(self):
        # Common irregular plurals
        self.irregular_plurals = {
            'children': 'child',
            'people': 'person',
            'men': 'man',
            'women': 'woman',
            'teeth': 'tooth',
            'feet': 'foot',
            'mice': 'mouse',
            'geese': 'goose',
            'phenomena': 'phenomenon',
            'criteria': 'criterion'
        }

        # Phrases to ignore
        self.ignore_phrases = [
            'thank you'
        ]
        
    def normalize_word(self, word: str) -> str:
        """Normalize a word to its singular form."""
        # Check irregular plurals first
        if word.lower() in self.irregular_plurals:
            return self.irregular_plurals[word.lower()]
            
        # Handle regular plurals
        if word.endswith('s'):
            # Words ending in 'ies'
            if word.endswith('ies'):
                if len(word) > 3:  # Avoid words like "ties"
                    return word[:-3] + 'y'
            # Words ending in 'es'
            elif word.endswith(('ses', 'zes', 'ches', 'shes')):
                return word[:-2]
            # Regular 's' plural
            else:
                return word[:-1]
                
        return word

    def remove_ignore_phrases(self, text: str) -> str:
        """Remove phrases that should be ignored from the text."""
        text_lower = text.lower()
        for phrase in self.ignore_phrases:
            text_lower = text_lower.replace(phrase, '')
        
        # Remove extra whitespace and normalize spaces
        return ' '.join(text_lower.split())

    def normalize_text(self, text: str) -> str:
        """Normalize text by removing ignore phrases and handling plurals."""
        # First remove ignored phrases
        text = self.remove_ignore_phrases(text)
        
        # Then normalize plurals
        words = text.split()
        normalized_words = [self.normalize_word(word) for word in words]
        return ' '.join(normalized_words)

class WERCalculator:
    def __init__(self, html_output: bool = True, ignore_insertions: bool = True):
        """Initialize WER calculator with HTML output and insertion handling options."""
        self.html_output = html_output
        self.ignore_insertions = ignore_insertions
        self.aligned_htmls: List[str] = []
        self.error_examples = {'sub': [], 'ins': [], 'del': []}
        self.wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': 0}
        self.html_generator = HTMLGenerator(ignore_insertions=ignore_insertions) if html_output else None
        self.plural_normalizer = PluralNormalizer()

    def reset_stats(self):
        """Reset all statistics counters."""
        self.wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': 0}
        self.error_examples = {'sub': [], 'ins': [], 'del': []}
        self.aligned_htmls = []

    def _compute_alignment_matrix(self, hyp_words: List[str], ref_words: List[str], 
                                norm_hyp_words: List[str], norm_ref_words: List[str]) -> Tuple[List[List], List[List]]:
        """Compute edit distance matrix with operation tracking."""
        m, n = len(ref_words), len(hyp_words)
        D = [[0] * (n + 1) for _ in range(m + 1)]
        ops = [[None] * (n + 1) for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            D[i][0] = i
            if i > 0:
                ops[i][0] = 'del'
        for j in range(n + 1):
            # Use insertion cost based on ignore_insertions setting
            D[0][j] = 0 if self.ignore_insertions else j
            if j > 0:
                ops[0][j] = 'ins'
        
        # Fill matrices using dynamic programming
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if norm_ref_words[i - 1] == norm_hyp_words[j - 1]:
                    D[i][j] = D[i - 1][j - 1]
                    ops[i][j] = 'equal'
                else:
                    sub_cost = D[i - 1][j - 1] + 1
                    ins_cost = D[i][j - 1] + (0 if self.ignore_insertions else 1)
                    del_cost = D[i - 1][j] + 1
                    
                    D[i][j] = min(sub_cost, ins_cost, del_cost)
                    
                    if D[i][j] == sub_cost:
                        ops[i][j] = 'sub'
                    elif D[i][j] == ins_cost:
                        ops[i][j] = 'ins'
                    else:
                        ops[i][j] = 'del'
        
        return D, ops

    def _generate_html(self, hyp: str, ref: str, err_type: str) -> str:
        """Generate HTML for word alignment visualization."""
        if err_type == 'equal':
            return f'{hyp} '
        
        # Skip insertion highlighting if ignoring insertions
        if err_type == 'ins' and self.ignore_insertions:
            return f'{hyp} '
            
        styles = {
            'sub': f'''<span style="background-color: yellow"><del>{hyp}</del></span>
                      <span style="background-color: yellow">{ref}</span> ''',
            'del': f'''<span style="background-color: red">{ref}</span> ''',
            'ins': f'''<span style="background-color: green"><del>{hyp}</del></span> '''
        }
        return styles.get(err_type, '')

    def compute_wer(self, hypothesis: str, reference: str) -> Dict:
        """Compute WER using improved alignment algorithm with configurable insertion handling."""
        # First apply thank you filtering and normalization
        hypothesis = self.plural_normalizer.remove_ignore_phrases(hypothesis)
        reference = self.plural_normalizer.remove_ignore_phrases(reference)
        
        # Split into words and normalize plurals
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        norm_ref_words = [self.plural_normalizer.normalize_word(w.lower()) for w in ref_words]
        norm_hyp_words = [self.plural_normalizer.normalize_word(w.lower()) for w in hyp_words]
        
        D, ops = self._compute_alignment_matrix(hyp_words, ref_words, norm_hyp_words, norm_ref_words)
        
        aligned_html = []
        temp_error_examples = {'sub': [], 'ins': [], 'del': []}
        self.wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': 0}
        
        i, j = len(ref_words), len(hyp_words)
        alignments = []
        
        while i > 0 or j > 0:
            op = ops[i][j]
            if op == 'equal':
                alignments.insert(0, ('equal', hyp_words[j-1], hyp_words[j-1]))
                i -= 1
                j -= 1
            elif op == 'sub':
                alignments.insert(0, ('sub', hyp_words[j-1], ref_words[i-1]))
                i -= 1
                j -= 1
            elif op == 'del':
                alignments.insert(0, ('del', "", ref_words[i-1]))
                i -= 1
            else:  # insertion
                alignments.insert(0, ('ins', hyp_words[j-1], ""))
                j -= 1
        
        for op, hyp_word, ref_word in alignments:
            if op == 'equal':
                aligned_html.append(self._generate_html(hyp_word, hyp_word, "equal"))
                self.wer_info['nw'] += 1
            elif op == 'sub':
                norm_hyp = self.plural_normalizer.normalize_word(hyp_word.lower())
                norm_ref = self.plural_normalizer.normalize_word(ref_word.lower())
                if norm_hyp == norm_ref:
                    aligned_html.append(self._generate_html(hyp_word, hyp_word, "equal"))
                    self.wer_info['nw'] += 1
                else:
                    aligned_html.append(self._generate_html(hyp_word, ref_word, "sub"))
                    temp_error_examples['sub'].append((hyp_word, ref_word))
                    self.wer_info['sub'] += 1
                    self.wer_info['nw'] += 1
            elif op == 'del':
                aligned_html.append(self._generate_html("", ref_word, "del"))
                temp_error_examples['del'].append(ref_word)
                self.wer_info['del'] += 1
                self.wer_info['nw'] += 1
            elif op == 'ins':
                aligned_html.append(self._generate_html(hyp_word, "", "ins"))
                if not self.ignore_insertions:
                    temp_error_examples['ins'].append(hyp_word)
                    self.wer_info['ins'] += 1
                    self.wer_info['nw'] += 1
        
        self.error_examples = temp_error_examples
        
        if self.html_output:
            self.aligned_htmls.append(''.join(aligned_html))
        
        return self._calculate_stats()

    def _calculate_stats(self) -> Dict:
        """Calculate WER statistics."""
        total_words = max(self.wer_info['nw'], 1)
        total_errors = self.wer_info['sub'] + self.wer_info['del']
        if not self.ignore_insertions:
            total_errors += self.wer_info['ins']
            
        wer = (total_errors * 100.0) / total_words
        
        stats = {
            'wer': wer,
            'accuracy': 100 - wer,
            'substitutions': (self.wer_info['sub'] * 100.0) / total_words,
            'deletions': (self.wer_info['del'] * 100.0) / total_words,
            'total_words': total_words,
            'total_errors': total_errors
        }
        
        # Only include insertions if not ignored
        if not self.ignore_insertions:
            stats['insertions'] = (self.wer_info['ins'] * 100.0) / total_words
        else:
            stats['insertions'] = 0.0
            
        return stats

    def save_html(self, filename: str) -> None:
        """Save HTML visualization using the HTML generator."""
        if not self.html_output or not self.html_generator:
            return
            
        stats = self._calculate_stats()
        error_counts = {
            'sub': len(self.error_examples['sub']),
            'del': len(self.error_examples['del']),
            'ins': len(self.error_examples['ins'])  # Keep track for visualization
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
    html_output: bool = typer.Option(True, "--html", help="Generate HTML visualization"),
    ignore_insertions: bool = typer.Option(False, "--ignore-insertions", help="Count insertions in WER calculation", is_flag=True)
):
    """Calculate Word Error Rate (WER) between prediction and reference files."""
    # Read and normalize files
    predictions = read_file(prediction_file, normalize, remove_comments)
    references = read_file(reference_file, normalize, remove_comments)
    
    print("\n" + "=" * 50)
    print(f"Configuration: normalize={normalize}, remove_comments={remove_comments}, ignore_insertions={ignore_insertions}")
    if ignore_insertions:
        print("Note: Insertions are ignored in WER calculation")
    
    # Initialize calculator
    calculator = WERCalculator(html_output=html_output, ignore_insertions=ignore_insertions)
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
        if not ignore_insertions:
            print(f"Insertions: {stats['insertions']:.1f}%")
        else:
            print("Insertions: Ignored")
        print(f"\nTotal Words: {stats['total_words']}")
        print(f"Total Errors: {stats['total_errors']}")
        if ignore_insertions:
            print("(excluding insertions)")
    
        if html_output:
            calculator.save_html(prediction_file)
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    app()