import fitz
doc = fitz.open("CBIC_ALL_PDFS/chap-29.pdf")
print("Total pages in chap-29.pdf:", doc.page_count)