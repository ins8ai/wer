import re
import sys
import typer
from evaluate import load
from whisper_norm import EnglishTextNormalizer
from typing import Optional

app = typer.Typer()
wer_metric = load("wer")
normalizer = EnglishTextNormalizer()

def TxtPreprocess(txt):
    """Preprocess text before WER calculation."""
    # Remove dollar signs
    txt = txt.replace('$', '')
    
    # Lowercase first
    txt = txt.lower()
    
    # Split into words, remove trailing 's' and rejoin
    words = txt.split()
    words = [word[:-1] if word.endswith('s') else word for word in words]
    txt = ' '.join(words)
    
    # Lowercase, remove \t and new line
    txt = re.sub(r'[\t\n]', ' ', txt)
    
    # Remove punctuation before space
    txt = re.sub(r'[,.?!]+ ', ' ', txt)
    
    # Remove punctuation before end
    txt = re.sub(r'[,.?!]+$', ' ', txt)
    
    # Remove punctuation after space
    txt = re.sub(r' [,.?!]+', ' ', txt)
    
    # Remove quotes, [, ], ( and )
    txt = re.sub(r'["()\[\]]', '', txt)
    
    # Remove extra space
    txt = re.sub(' +', ' ', txt.strip())
    return txt

def RemoveCommentTxtPreprocess(txt):
    """Preprocess text and remove comments in the bracket, such as [comments]."""
    # Remove comments surrounded by box brackets:
    txt = re.sub(r'\[\w+\]', '', txt)
    return TxtPreprocess(txt)

def read_file(file_path: str, normalize: bool = False, remove_comments: bool = False):
    """Read and process file contents."""
    lines = []
    with open(file_path, "r") as file:
        for line in file:
            text = line.strip()
            if normalize:
                text = normalizer(text)
            if remove_comments:
                text = RemoveCommentTxtPreprocess(text)
            lines.append(text)
    return lines

def highlight_aligned_html(hyp, ref, err_type):
    """Generate a html element to highlight the difference between hyp and ref."""
    highlighted_html = ''
    if err_type == 'none':
        if hyp != ref:
            raise ValueError('hyp (%s) does not match ref (%s) for none error' % (hyp, ref))
        highlighted_html += '%s ' % hyp
    elif err_type == 'sub':
        highlighted_html += """<span style="background-color: yellow">
            <del>%s</del></span><span style="background-color: yellow">
            %s </span> """ % (hyp, ref)
    elif err_type == 'del':
        highlighted_html += """<span style="background-color: red">
            %s </span> """ % (ref)
    elif err_type == 'ins':
        highlighted_html += """<span style="background-color: green">
            <del>%s</del> </span> """ % (hyp)
    else:
        raise ValueError('unknown err_type ' + err_type)
    return highlighted_html

def compute_edit_distance_matrix(hyp_words, ref_words):
    """Compute edit distance between two list of strings."""
    reference_length_plus = len(ref_words) + 1
    hypothesis_length_plus = len(hyp_words) + 1
    edit_dist_mat = [[]] * reference_length_plus

    # Initialization.
    for i in range(reference_length_plus):
        edit_dist_mat[i] = [0] * hypothesis_length_plus
        for j in range(hypothesis_length_plus):
            if i == 0:
                edit_dist_mat[0][j] = j
            elif j == 0:
                edit_dist_mat[i][0] = i

    # Do dynamic programming.
    for i in range(1, reference_length_plus):
        for j in range(1, hypothesis_length_plus):
            if ref_words[i - 1] == hyp_words[j - 1]:
                edit_dist_mat[i][j] = edit_dist_mat[i - 1][j - 1]
            else:
                tmp0 = edit_dist_mat[i - 1][j - 1] + 1
                tmp1 = edit_dist_mat[i][j - 1] + 1
                tmp2 = edit_dist_mat[i - 1][j] + 1
                edit_dist_mat[i][j] = min(tmp0, tmp1, tmp2)

    return edit_dist_mat

class WERCalculator:
    def __init__(self, html_output: bool = True):
        self.html_output = html_output
        self.aligned_htmls = []
        self.wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': 0}

    def compute_wer(self, hypothesis: str, reference: str):
        """Compute WER and generate alignment visualization."""
        hyp_words = hypothesis.split()
        ref_words = reference.split()
        
        # Compute edit distance
        distmat = compute_edit_distance_matrix(hyp_words, ref_words)
        
        # Back trace, to distinguish different errors: ins, del, sub.
        pos_hyp, pos_ref = len(hyp_words), len(ref_words)
        aligned_html = ''
        
        while pos_hyp > 0 or pos_ref > 0:
            err_type = ''
            
            # Distinguish error type by back tracking
            if pos_ref == 0:
                err_type = 'ins'
            elif pos_hyp == 0:
                err_type = 'del'
            else:
                if hyp_words[pos_hyp - 1] == ref_words[pos_ref - 1]:
                    err_type = 'none'  # correct error
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp - 1] + 1:
                    err_type = 'sub'  # substitute error
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp] + 1:
                    err_type = 'del'  # deletion error
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref][pos_hyp - 1] + 1:
                    err_type = 'ins'  # insertion error
                else:
                    raise ValueError('fail to parse edit distance matrix.')

            # Generate aligned_html with the original highlighting style
            if self.html_output:
                hyp_word = ' ' if pos_hyp == 0 else hyp_words[pos_hyp - 1]
                ref_word = ' ' if pos_ref == 0 else ref_words[pos_ref - 1]
                aligned_html = highlight_aligned_html(hyp_word, ref_word, err_type) + aligned_html

            # Update error counts
            if err_type != 'none':
                self.wer_info[err_type] += 1
            if err_type != 'ins':
                self.wer_info['nw'] += 1

            # Adjust position based on error type
            if err_type == 'del':
                pos_ref = pos_ref - 1
            elif err_type == 'ins':
                pos_hyp = pos_hyp - 1
            else:  # err_type in ['sub', 'none']
                pos_hyp, pos_ref = pos_hyp - 1, pos_ref - 1
        
        if self.html_output:
            self.aligned_htmls.append(aligned_html)
        
        return self.get_wer_stats()

    def get_wer_stats(self):
        """Get WER statistics."""
        total_words = max(self.wer_info['nw'], 1)
        total_errors = self.wer_info['sub'] + self.wer_info['del'] + self.wer_info['ins']
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

    def save_html(self, filename: str):
        """Save alignment visualization to HTML file."""
        if self.html_output:
            try:
                html_filename = f'{filename}_diagnosis.html'
                print(f"\nAttempting to save HTML to: {html_filename}")
                print(f"Number of aligned segments: {len(self.aligned_htmls)}")
                
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write('<!DOCTYPE html>\n<html>\n<head>\n')
                    f.write('<meta charset="utf-8">\n')
                    f.write('<title>WER Analysis</title>\n')
                    f.write('</head>\n<body>\n')
                    f.write('<div style="font-family: Arial, sans-serif; line-height: 1.6;">\n')
                    for i, html in enumerate(self.aligned_htmls):
                        f.write(f'<p>Segment {i+1}:</p>\n')
                        f.write(f'<div style="margin-bottom: 20px;">{html}</div>\n')
                    f.write('</div>\n</body>\n</html>')
                
                print(f"Successfully saved HTML file to: {html_filename}")
            except Exception as e:
                print(f'Failed to write diagnosis HTML file: {str(e)}')

@app.command()
def main(
    prediction_file: str,
    reference_file: str,
    normalize: bool = typer.Option(True, "--normalize", "-n", help="Apply text normalization"),
    remove_comments: bool = typer.Option(True, "--remove-comments", "-r", help="Remove comments in brackets"),
    html_output: bool = typer.Option(True, "--html", help="Generate HTML visualization"),
    huggingface_wer: bool = typer.Option(True, "--hf", help="Use HuggingFace WER calculation")
):
    """
    Calculate Word Error Rate (WER) between prediction and reference files.
    """
    # Read and process files
    predictions = read_file(prediction_file, normalize, remove_comments)
    references = read_file(reference_file, normalize, remove_comments)

    print("\n" + "=" * 50)
    print(f"Normalization: {normalize}")
    print(f"Remove Comments: {remove_comments}")
    
    if huggingface_wer:
        # Use HuggingFace WER calculation
        wer_score = wer_metric.compute(predictions=predictions, references=references)
        accuracy = (1.0 - wer_score) * 100
        print(f"Word Error Rate (HuggingFace): {wer_score:.3f}")
        print(f"Accuracy: {accuracy:.1f}%")
    
    # Always do detailed WER calculation for HTML visualization
        calculator = WERCalculator(html_output=html_output)
        
        print("\nProcessing files:")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Number of references: {len(references)}")
        
        for i, (hyp, ref) in enumerate(zip(predictions, references)):
            print(f"\nProcessing segment {i+1}:")
            print(f"Hypothesis: {hyp[:50]}...")
            print(f"Reference:  {ref[:50]}...")
            stats = calculator.compute_wer(hyp, ref)
        
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
            print(f"\nHTML diagnosis saved to: {prediction_file}_diagnosis.html")
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    app()

def RemoveCommentTxtPreprocess(txt):
    """Preprocess text and remove comments in the bracket, such as [comments]."""
    # Remove comments surrounded by box brackets:
    txt = re.sub(r'\[\w+\]', '', txt)
    return TxtPreprocess(txt)

def read_file(file_path: str, normalize: bool = False, remove_comments: bool = False):
    """Read and process file contents."""
    lines = []
    with open(file_path, "r") as file:
        for line in file:
            text = line.strip()
            if normalize:
                text = normalizer(text)
            if remove_comments:
                text = RemoveCommentTxtPreprocess(text)
            lines.append(text)
    return lines

def highlight_aligned_html(hyp, ref, err_type):
    """Generate a html element to highlight the difference between hyp and ref."""
    highlighted_html = ''
    if err_type == 'none':
        if hyp != ref:
            raise ValueError('hyp (%s) does not match ref (%s) for none error' % (hyp, ref))
        highlighted_html += '%s ' % hyp
    elif err_type == 'sub':
        highlighted_html += """<span style="background-color: yellow">
            <del>%s</del></span><span style="background-color: yellow">
            %s </span> """ % (hyp, ref)
    elif err_type == 'del':
        highlighted_html += """<span style="background-color: red">
            %s </span> """ % (ref)
    elif err_type == 'ins':
        highlighted_html += """<span style="background-color: green">
            <del>%s</del> </span> """ % (hyp)
    else:
        raise ValueError('unknown err_type ' + err_type)
    return highlighted_html

def compute_edit_distance_matrix(hyp_words, ref_words):
    """Compute edit distance between two list of strings."""
    reference_length_plus = len(ref_words) + 1
    hypothesis_length_plus = len(hyp_words) + 1
    edit_dist_mat = [[]] * reference_length_plus

    # Initialization.
    for i in range(reference_length_plus):
        edit_dist_mat[i] = [0] * hypothesis_length_plus
        for j in range(hypothesis_length_plus):
            if i == 0:
                edit_dist_mat[0][j] = j
            elif j == 0:
                edit_dist_mat[i][0] = i

    # Do dynamic programming.
    for i in range(1, reference_length_plus):
        for j in range(1, hypothesis_length_plus):
            if ref_words[i - 1] == hyp_words[j - 1]:
                edit_dist_mat[i][j] = edit_dist_mat[i - 1][j - 1]
            else:
                tmp0 = edit_dist_mat[i - 1][j - 1] + 1
                tmp1 = edit_dist_mat[i][j - 1] + 1
                tmp2 = edit_dist_mat[i - 1][j] + 1
                edit_dist_mat[i][j] = min(tmp0, tmp1, tmp2)

    return edit_dist_mat

class WERCalculator:
    def __init__(self, html_output: bool = True):
        self.html_output = html_output
        self.aligned_htmls = []
        self.wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': 0}

    def compute_wer(self, hypothesis: str, reference: str):
        """Compute WER and generate alignment visualization."""
        hyp_words = hypothesis.split()
        ref_words = reference.split()
        
        # Compute edit distance
        distmat = compute_edit_distance_matrix(hyp_words, ref_words)
        
        # Back trace, to distinguish different errors: ins, del, sub.
        pos_hyp, pos_ref = len(hyp_words), len(ref_words)
        aligned_html = ''
        
        while pos_hyp > 0 or pos_ref > 0:
            err_type = ''
            
            # Distinguish error type by back tracking
            if pos_ref == 0:
                err_type = 'ins'
            elif pos_hyp == 0:
                err_type = 'del'
            else:
                if hyp_words[pos_hyp - 1] == ref_words[pos_ref - 1]:
                    err_type = 'none'  # correct error
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp - 1] + 1:
                    err_type = 'sub'  # substitute error
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp] + 1:
                    err_type = 'del'  # deletion error
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref][pos_hyp - 1] + 1:
                    err_type = 'ins'  # insertion error
                else:
                    raise ValueError('fail to parse edit distance matrix.')

            # Generate aligned_html with the original highlighting style
            if self.html_output:
                hyp_word = ' ' if pos_hyp == 0 else hyp_words[pos_hyp - 1]
                ref_word = ' ' if pos_ref == 0 else ref_words[pos_ref - 1]
                aligned_html = highlight_aligned_html(hyp_word, ref_word, err_type) + aligned_html

            # Update error counts
            if err_type != 'none':
                self.wer_info[err_type] += 1
            if err_type != 'ins':
                self.wer_info['nw'] += 1

            # Adjust position based on error type
            if err_type == 'del':
                pos_ref = pos_ref - 1
            elif err_type == 'ins':
                pos_hyp = pos_hyp - 1
            else:  # err_type in ['sub', 'none']
                pos_hyp, pos_ref = pos_hyp - 1, pos_ref - 1
        
        if self.html_output:
            self.aligned_htmls.append(aligned_html)
        
        return self.get_wer_stats()

    def get_wer_stats(self):
        """Get WER statistics."""
        total_words = max(self.wer_info['nw'], 1)
        total_errors = self.wer_info['sub'] + self.wer_info['del'] + self.wer_info['ins']
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

    def save_html(self, filename: str):
        """Save alignment visualization to HTML file."""
        if self.html_output:
            try:
                html_filename = f'{filename}_diagnosis.html'
                print(f"\nAttempting to save HTML to: {html_filename}")
                print(f"Number of aligned segments: {len(self.aligned_htmls)}")
                
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write('<!DOCTYPE html>\n<html>\n<head>\n')
                    f.write('<meta charset="utf-8">\n')
                    f.write('<title>WER Analysis</title>\n')
                    f.write('</head>\n<body>\n')
                    f.write('<div style="font-family: Arial, sans-serif; line-height: 1.6;">\n')
                    for i, html in enumerate(self.aligned_htmls):
                        f.write(f'<p>Segment {i+1}:</p>\n')
                        f.write(f'<div style="margin-bottom: 20px;">{html}</div>\n')
                    f.write('</div>\n</body>\n</html>')
                
                print(f"Successfully saved HTML file to: {html_filename}")
            except Exception as e:
                print(f'Failed to write diagnosis HTML file: {str(e)}')

@app.command()
def main(
    prediction_file: str,
    reference_file: str,
    normalize: bool = typer.Option(True, "--normalize", "-n", help="Apply text normalization"),
    remove_comments: bool = typer.Option(True, "--remove-comments", "-r", help="Remove comments in brackets"),
    html_output: bool = typer.Option(True, "--html", help="Generate HTML visualization"),
    huggingface_wer: bool = typer.Option(True, "--hf", help="Use HuggingFace WER calculation")
):
    """
    Calculate Word Error Rate (WER) between prediction and reference files.
    """
    # Read and process files
    predictions = read_file(prediction_file, normalize, remove_comments)
    references = read_file(reference_file, normalize, remove_comments)

    print("\n" + "=" * 50)
    print(f"Normalization: {normalize}")
    print(f"Remove Comments: {remove_comments}")
    
    if huggingface_wer:
        # Use HuggingFace WER calculation
        wer_score = wer_metric.compute(predictions=predictions, references=references)
        accuracy = (1.0 - wer_score) * 100
        print(f"Word Error Rate (HuggingFace): {wer_score:.3f}")
        print(f"Accuracy: {accuracy:.1f}%")
    
    # Always do detailed WER calculation for HTML visualization
        calculator = WERCalculator(html_output=html_output)
        
        print("\nProcessing files:")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Number of references: {len(references)}")
        
        for i, (hyp, ref) in enumerate(zip(predictions, references)):
            print(f"\nProcessing segment {i+1}:")
            print(f"Hypothesis: {hyp[:50]}...")
            print(f"Reference:  {ref[:50]}...")
            stats = calculator.compute_wer(hyp, ref)
        
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
            print(f"\nHTML diagnosis saved to: {prediction_file}_diagnosis.html")
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    app()

def RemoveCommentTxtPreprocess(txt):
    """Preprocess text and remove comments in the bracket, such as [comments]."""
    # Remove comments surrounded by box brackets:
    txt = re.sub(r'\[\w+\]', '', txt)
    return TxtPreprocess(txt)

def read_file(file_path: str, normalize: bool = False, remove_comments: bool = False):
    """Read and process file contents."""
    lines = []
    with open(file_path, "r") as file:
        for line in file:
            text = line.strip()
            if normalize:
                text = normalizer(text)
            if remove_comments:
                text = RemoveCommentTxtPreprocess(text)
            lines.append(text)
    return lines

def highlight_aligned_html(hyp, ref, err_type):
    """Generate a html element to highlight the difference between hyp and ref."""
    highlighted_html = ''
    if err_type == 'none':
        if hyp != ref:
            raise ValueError('hyp (%s) does not match ref (%s) for none error' % (hyp, ref))
        highlighted_html += '%s ' % hyp
    elif err_type == 'sub':
        highlighted_html += """<span style="background-color: yellow">
            <del>%s</del></span><span style="background-color: yellow">
            %s </span> """ % (hyp, ref)
    elif err_type == 'del':
        highlighted_html += """<span style="background-color: red">
            %s </span> """ % (ref)
    elif err_type == 'ins':
        highlighted_html += """<span style="background-color: green">
            <del>%s</del> </span> """ % (hyp)
    else:
        raise ValueError('unknown err_type ' + err_type)
    return highlighted_html

def compute_edit_distance_matrix(hyp_words, ref_words):
    """Compute edit distance between two list of strings."""
    reference_length_plus = len(ref_words) + 1
    hypothesis_length_plus = len(hyp_words) + 1
    edit_dist_mat = [[]] * reference_length_plus

    # Initialization.
    for i in range(reference_length_plus):
        edit_dist_mat[i] = [0] * hypothesis_length_plus
        for j in range(hypothesis_length_plus):
            if i == 0:
                edit_dist_mat[0][j] = j
            elif j == 0:
                edit_dist_mat[i][0] = i

    # Do dynamic programming.
    for i in range(1, reference_length_plus):
        for j in range(1, hypothesis_length_plus):
            if ref_words[i - 1] == hyp_words[j - 1]:
                edit_dist_mat[i][j] = edit_dist_mat[i - 1][j - 1]
            else:
                tmp0 = edit_dist_mat[i - 1][j - 1] + 1
                tmp1 = edit_dist_mat[i][j - 1] + 1
                tmp2 = edit_dist_mat[i - 1][j] + 1
                edit_dist_mat[i][j] = min(tmp0, tmp1, tmp2)

    return edit_dist_mat

class WERCalculator:
    def __init__(self, html_output: bool = True):
        self.html_output = html_output
        self.aligned_htmls = []
        self.wer_info = {'sub': 0, 'ins': 0, 'del': 0, 'nw': 0}

    def compute_wer(self, hypothesis: str, reference: str):
        """Compute WER and generate alignment visualization."""
        hyp_words = hypothesis.split()
        ref_words = reference.split()
        
        # Compute edit distance
        distmat = compute_edit_distance_matrix(hyp_words, ref_words)
        
        # Back trace, to distinguish different errors: ins, del, sub.
        pos_hyp, pos_ref = len(hyp_words), len(ref_words)
        aligned_html = ''
        
        while pos_hyp > 0 or pos_ref > 0:
            err_type = ''
            
            # Distinguish error type by back tracking
            if pos_ref == 0:
                err_type = 'ins'
            elif pos_hyp == 0:
                err_type = 'del'
            else:
                if hyp_words[pos_hyp - 1] == ref_words[pos_ref - 1]:
                    err_type = 'none'  # correct error
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp - 1] + 1:
                    err_type = 'sub'  # substitute error
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref - 1][pos_hyp] + 1:
                    err_type = 'del'  # deletion error
                elif distmat[pos_ref][pos_hyp] == distmat[pos_ref][pos_hyp - 1] + 1:
                    err_type = 'ins'  # insertion error
                else:
                    raise ValueError('fail to parse edit distance matrix.')

            # Generate aligned_html with the original highlighting style
            if self.html_output:
                hyp_word = ' ' if pos_hyp == 0 else hyp_words[pos_hyp - 1]
                ref_word = ' ' if pos_ref == 0 else ref_words[pos_ref - 1]
                aligned_html = highlight_aligned_html(hyp_word, ref_word, err_type) + aligned_html

            # Update error counts
            if err_type != 'none':
                self.wer_info[err_type] += 1
            if err_type != 'ins':
                self.wer_info['nw'] += 1

            # Adjust position based on error type
            if err_type == 'del':
                pos_ref = pos_ref - 1
            elif err_type == 'ins':
                pos_hyp = pos_hyp - 1
            else:  # err_type in ['sub', 'none']
                pos_hyp, pos_ref = pos_hyp - 1, pos_ref - 1
        
        if self.html_output:
            self.aligned_htmls.append(aligned_html)
        
        return self.get_wer_stats()

    def get_wer_stats(self):
        """Get WER statistics."""
        total_words = max(self.wer_info['nw'], 1)
        total_errors = self.wer_info['sub'] + self.wer_info['del'] + self.wer_info['ins']
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

    def save_html(self, filename: str):
        """Save alignment visualization to HTML file."""
        if self.html_output:
            try:
                html_filename = f'{filename}_diagnosis.html'
                print(f"\nAttempting to save HTML to: {html_filename}")
                print(f"Number of aligned segments: {len(self.aligned_htmls)}")
                
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write('<!DOCTYPE html>\n<html>\n<head>\n')
                    f.write('<meta charset="utf-8">\n')
                    f.write('<title>WER Analysis</title>\n')
                    f.write('</head>\n<body>\n')
                    f.write('<div style="font-family: Arial, sans-serif; line-height: 1.6;">\n')
                    for i, html in enumerate(self.aligned_htmls):
                        f.write(f'<p>Segment {i+1}:</p>\n')
                        f.write(f'<div style="margin-bottom: 20px;">{html}</div>\n')
                    f.write('</div>\n</body>\n</html>')
                
                print(f"Successfully saved HTML file to: {html_filename}")
            except Exception as e:
                print(f'Failed to write diagnosis HTML file: {str(e)}')

@app.command()
def main(
    prediction_file: str,
    reference_file: str,
    normalize: bool = typer.Option(True, "--normalize", "-n", help="Apply text normalization"),
    remove_comments: bool = typer.Option(True, "--remove-comments", "-r", help="Remove comments in brackets"),
    html_output: bool = typer.Option(True, "--html", help="Generate HTML visualization"),
    huggingface_wer: bool = typer.Option(True, "--hf", help="Use HuggingFace WER calculation")
):
    """
    Calculate Word Error Rate (WER) between prediction and reference files.
    """
    # Read and process files
    predictions = read_file(prediction_file, normalize, remove_comments)
    references = read_file(reference_file, normalize, remove_comments)

    print("\n" + "=" * 50)
    print(f"Normalization: {normalize}")
    print(f"Remove Comments: {remove_comments}")
    
    if huggingface_wer:
        # Use HuggingFace WER calculation
        wer_score = wer_metric.compute(predictions=predictions, references=references)
        accuracy = (1.0 - wer_score) * 100
        print(f"Word Error Rate (HuggingFace): {wer_score:.3f}")
        print(f"Accuracy: {accuracy:.1f}%")
    
    # Always do detailed WER calculation for HTML visualization
        calculator = WERCalculator(html_output=html_output)
        
        print("\nProcessing files:")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Number of references: {len(references)}")
        
        for i, (hyp, ref) in enumerate(zip(predictions, references)):
            print(f"\nProcessing segment {i+1}:")
            print(f"Hypothesis: {hyp[:50]}...")
            print(f"Reference:  {ref[:50]}...")
            stats = calculator.compute_wer(hyp, ref)
        
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
            print(f"\nHTML diagnosis saved to: {prediction_file}_diagnosis.html")
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    app()