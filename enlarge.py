import re

with open('paper/fig_architecture.tex', 'r') as f:
    content = f.read()

content = content.replace(r'\small', r'\large')
content = content.replace(r'font=\sffamily\large,', r'font=\sffamily\Large,')
content = content.replace(r'\large\bfseries', r'\Large\bfseries')

content = content.replace('line width=0.7pt', 'line width=1.2pt')
content = content.replace('line width=0.6pt', 'line width=1.0pt')
content = content.replace('line width=0.5pt', 'line width=0.8pt')
content = content.replace('line width=0.8pt', 'line width=1.5pt')

with open('paper/fig_architecture.tex', 'w') as f:
    f.write(content)
