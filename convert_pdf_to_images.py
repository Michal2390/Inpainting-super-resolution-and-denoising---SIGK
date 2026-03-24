#!/usr/bin/env python3
"""
Konwersja PDF'a na obrazy PNG za pomocą PyMuPDF
"""

import fitz  # PyMuPDF
import os
from pathlib import Path

# Ścieżka do PDF'a
pdf_path = r"data/SIGK_1___final (1).pdf"
output_dir = "data/report_images"

# Utwórz folder na obrazy
Path(output_dir).mkdir(parents=True, exist_ok=True)

print(f"Konwersja PDF: {pdf_path}")
print(f"Folder wyjściowy: {output_dir}")

try:
    # Otwórz PDF
    pdf_document = fitz.open(pdf_path)
    
    print(f"\nOdkryto {len(pdf_document)} stron")
    
    # Konwertuj każdą stronę na obraz
    for page_num in range(len(pdf_document)):
        # Renderuj stronę na obraz
        page = pdf_document[page_num]
        
        # Zwiększ zoom dla lepszej jakości (1.5x)
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        
        output_path = os.path.join(output_dir, f"page_{page_num + 1:02d}.png")
        pix.save(output_path)
        print(f"✅ Zapisano: {output_path}")
    
    pdf_document.close()
    
    print("\n✨ Konwersja zakończona!")
    print(f"Obrazy znajdują się w: {output_dir}")
    
except Exception as e:
    print(f"❌ Błąd: {e}")
    import traceback
    traceback.print_exc()


