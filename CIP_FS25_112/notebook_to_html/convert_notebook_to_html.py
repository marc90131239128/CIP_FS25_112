import nbformat
from nbconvert import HTMLExporter
import os

def convert_notebook_to_html(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Create HTML exporter
    html_exporter = HTMLExporter()
    
    # Convert notebook to HTML
    html_output, resources = html_exporter.from_notebook_node(notebook)
    
    # Write the HTML file
    output_file = os.path.splitext(notebook_path)[0] + '.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"Conversion complete! HTML file saved as: {output_file}")

if __name__ == '__main__':
    convert_notebook_to_html('project_CIP_notebook.ipynb') 