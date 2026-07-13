import os
import re

filepath = r'c:\Users\harsh\Desktop\regul_ai - Copy\chat_bot\static\index.html'
stylepath = r'c:\Users\harsh\Desktop\regul_ai - Copy\chat_bot\static\style.css'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

style_match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
if style_match:
    css_content = style_match.group(1).strip()
    with open(stylepath, 'w', encoding='utf-8') as f:
        f.write(css_content)
    
    new_content = content[:style_match.start()] + '<link rel=\"stylesheet\" href=\"style.css\" />' + content[style_match.end():]
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('CSS extracted and HTML updated')
else:
    print('No style tag found')
