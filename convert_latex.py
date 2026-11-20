# -*- coding: utf-8 -*-
import sys

with open('report/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Change documentclass
content = content.replace(
    r'\documentclass[conference,a4paper]{IEEEtran}',
    r'''\documentclass[12pt,a4paper]{article}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{setspace}
\setstretch{1.2}'''
)

# 2. Change author block
old_author = r'''\author{
\IEEEauthorblockN{Phan Hồng Đạt}
\IEEEauthorblockA{
Khoa Mạng máy tính và Truyền thông\\
Trường Đại học Công nghệ Thông tin -- ĐHQG TP.HCM\\
Email: dat.ph@example.edu.vn
}
}'''

new_author = r'''\author{
\textbf{Phan Hồng Đạt}\\
Khoa Mạng máy tính và Truyền thông\\
Trường Đại học Công nghệ Thông tin -- ĐHQG TP.HCM\\
Email: dat.ph@example.edu.vn
}'''
content = content.replace(old_author, new_author)

# 3. Change abstract (IEEEtran puts it outside \maketitle, article puts it after)
# We don't necessarily need to move abstract, but abstract in article usually goes after \maketitle
# Wait, IEEEtran uses \begin{abstract} too, which is same in article.

# 4. Change keywords
content = content.replace(r'\begin{IEEEkeywords}', r'\textbf{Từ khóa:} ')
content = content.replace(r'\end{IEEEkeywords}', '')

with open('report/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print('Converted LaTeX successfully')
