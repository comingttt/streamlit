#!/usr/bin/env python
"""Helper script to write app.py with correct UTF-8 encoding"""
import sys

filepath = r'd:\homework\imghw\hw6\app.py'

content = """\
"""
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('Written successfully')
