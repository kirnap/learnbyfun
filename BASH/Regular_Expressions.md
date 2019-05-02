# Reqular expressions guideline
---
I always have some trouble to write down a regular expression for a given matching task. The thing though, nobody properly introduced what is the working principle behind regular expressions. Finally I found very very useful youtube video, I'd highly recommend to watch it for those who wants to be more competent on regular expressions.

1. [Video link](https://www.youtube.com/watch?v=sa-TUpSx1JA). Please do follow this video carefully.

Cheat Sheet
```

.       - Any Character Except New Line
\d      - Digit (0-9)
\D      - Not a Digit (0-9)
\w      - Word Character (a-z, A-Z, 0-9, _)
\W      - Not a Word Character
\s      - Whitespace (space, tab, newline)
\S      - Not Whitespace (space, tab, newline)

\b      - Word Boundary
\B      - Not a Word Boundary
^       - Beginning of a String
$       - End of a String

[]      - Matches Characters in brackets
[^ ]    - Matches Characters NOT in brackets
|       - Either Or
( )     - Group

Quantifiers:
*       - 0 or More
+       - 1 or More
?       - 0 or One
{3}     - Exact Number
{3,4}   - Range of Numbers (Minimum, Maximum)


#### Sample Regexs ####

[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+

```
