### Personal Blogging Notes
---
For now I am working on jekyll supported gh-pages tools, all the notes are related with it


---
This is how we done latex formula embedding to readme file: 
1. Suppose you are given with the following latex piece : 
`\begin{split} A & = \frac{\pi r^2}{2} \\  & = \frac{1}{2} \pi r^2 \end{split}`.
2. Go to 
https://alexanderrodin.com/github-latex-markdown/
3. Copy the formula above to the bar in that website
4. Copy the output from that website into your .md file (see raw version of this file to see how it works):

![\begin{split} A & = \frac{\pi r^2}{2} \\  & = \frac{1}{2} \pi r^2 \end{split}](https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%20A%20%26%20%3D%20%5Cfrac%7B%5Cpi%20r%5E2%7D%7B2%7D%20%5C%5C%20%20%26%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Cpi%20r%5E2%20%5Cend%7Bsplit%7D)

Read: [this link](https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b) for further details
