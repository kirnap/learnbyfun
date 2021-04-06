"""
Given a string s, return the longest palindromic substring in s.
"""


class Solution(object):
    def longestPalindrome(self, s:str) -> str:  # dynamic programming
        # 1. create table, make diagTrue
        mytab = []
        for i in range(len(s)):
            t = [False] * len(s)
            t[i] = True
            mytab.append(t)

        # 2.




    def longestPalindrome1(self, s: str) -> str:  # valid solution but takes too much time
        def isPalindrome(s:str) -> bool:
            start = 0
            endx = len(s)-1

            while start < endx:
                if s[start] != s[endx]:
                    return False
                start += 1
                endx -= 1
            return True
        longest = 0
        pals = s[0]
        for i in range(len(s)):
            for j in range(len(s)-1, -1, -1):
                if isPalindrome(s[i:j+1]):
                    wlen = j-i+1
                    if wlen > longest:
                        longest = wlen
                        pals = s[i:j+1]
        return pals


if __name__ == '__main__':
    s = Solution()
    mystring = 'aaaaaaa'
    print(s.longestPalindrome(mystring))








