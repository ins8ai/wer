"""html_generator.py: Module for generating HTML visualizations of WER analysis."""

from typing import Dict, List

class HTMLGenerator:
    def __init__(self):
        self.css = self._get_css()
    
    def _get_css(self) -> str:
        """Define CSS styles for the HTML output."""
        return '''
            body { 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                margin: 2em;
                color: #333;
            }
            .stats-summary {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
                border: 1px solid #dee2e6;
            }
            .stat-box {
                display: inline-block;
                padding: 15px 25px;
                margin: 10px;
                background-color: white;
                border-radius: 5px;
                border: 1px solid #dee2e6;
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin: 5px 0;
            }
            .stat-label {
                font-size: 14px;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .error-summary { 
                background-color: #f5f5f5; 
                padding: 20px; 
                border-radius: 5px;
                margin-top: 30px;
            }
            .error-type { margin-bottom: 20px; }
            .error-example { 
                display: inline-block;
                margin: 5px;
                padding: 5px 10px;
                background-color: #fff;
                border-radius: 3px;
                border: 1px solid #ddd;
            }
            .explanation { 
                background-color: #e9f5ff;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .error-count {
                color: #666;
                font-size: 0.9em;
                margin-left: 10px;
            }
        '''

    def _generate_header(self) -> str:
        """Generate HTML header section."""
        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>WER Analysis</title>
    <style>{self.css}</style>
</head>
<body>'''

    def _generate_stats_section(self, stats: Dict) -> str:
        """Generate statistics summary section."""
        return f'''
    <div class="stats-summary">
        <h2>Overall Performance</h2>
        <div class="stat-box">
            <div class="stat-label">Word Error Rate</div>
            <div class="stat-value">{stats['wer']:.1f}%</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Accuracy</div>
            <div class="stat-value">{stats['accuracy']:.1f}%</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Total Words</div>
            <div class="stat-value">{stats['total_words']}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Total Errors</div>
            <div class="stat-value">{stats['total_errors']}</div>
        </div>
    </div>'''

    def _generate_explanation(self) -> str:
        """Generate explanation section."""
        return '''
    <div class="explanation">
        <h2>Understanding Word Error Rate (WER)</h2>
        <p>WER measures how accurately text was transcribed by comparing it to a reference text. 
           There are three types of errors:</p>
        <ul>
            <li><strong>Substitutions</strong> (yellow): Words that were replaced with different words</li>
            <li><strong>Deletions</strong> (red): Words that were in the reference but missing from the transcription</li>
            <li><strong>Insertions</strong> (green): Extra words that appeared in the transcription but weren't in the reference</li>
        </ul>
    </div>'''

    def _generate_segments(self, aligned_htmls: List[str]) -> str:
        """Generate segments section."""
        segments = []
        for i, html in enumerate(aligned_htmls, 1):
            segments.append(f'<h3>Segment {i}:</h3>\n<div style="margin-bottom: 20px;">{html}</div>')
        return '\n'.join(segments)

    def _generate_error_summary(self, error_examples: Dict, error_counts: Dict[str, int]) -> str:
        """Generate error summary section."""
        error_sections = [
            ('sub', 'Substitutions (Incorrect → Correct)'),
            ('del', 'Deletions (Missing Words)'),
            ('ins', 'Insertions (Extra Words)')
        ]
        
        sections = ['<div class="error-summary">\n<h2>Error Summary</h2>']
        
        for err_type, title in error_sections:
            if error_examples[err_type]:
                count = error_counts[err_type]
                sections.append(f'<div class="error-type">')
                sections.append(f'<h3>{title}<span class="error-count">({count} words)</span></h3>')
                
                if err_type == 'sub':
                    sections.extend(
                        f'<span class="error-example">{hyp} → {ref}</span>'
                        for hyp, ref in reversed(error_examples[err_type])
                    )
                else:
                    sections.extend(
                        f'<span class="error-example">{word}</span>'
                        for word in reversed(error_examples[err_type])
                    )
                sections.append('</div>')
        
        sections.append('</div>')
        return '\n'.join(sections)

    def save_html(self, 
                 filename: str,
                 stats: Dict,
                 aligned_htmls: List[str],
                 error_examples: Dict,
                 error_counts: Dict[str, int]) -> None:
        """Generate and save the complete HTML visualization."""
        try:
            html_filename = f'{filename}_diagnosis.html'
            print(f"\nGenerating HTML visualization: {html_filename}")
            
            html_parts = [
                self._generate_header(),
                self._generate_stats_section(stats),
                self._generate_explanation(),
                self._generate_segments(aligned_htmls),
                self._generate_error_summary(error_examples, error_counts),
                '</body>\n</html>'
            ]
            
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_parts))
            
            print(f"HTML visualization saved successfully")
            
        except Exception as e:
            print(f'Failed to generate HTML visualization: {str(e)}')