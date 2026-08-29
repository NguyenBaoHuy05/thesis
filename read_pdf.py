import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

import pypdf

reader = pypdf.PdfReader(r'd:\Thesis\New folder\2304.14068v2.pdf')
print(f'Pages: {len(reader.pages)}', flush=True)

output_file = r'd:\Thesis\dcr_paper_text.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for i in range(len(reader.pages)):
        f.write(f'--- PAGE {i+1} ---\n')
        text = reader.pages[i].extract_text()
        if text:
            f.write(text)
        f.write('\n\n')

print(f'Text saved to {output_file}', flush=True)
