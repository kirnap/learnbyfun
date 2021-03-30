# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         curr = ''
#         longest = 0
#         for l in s:
#             if l in curr:
#                 if len(curr) > longest:
#                     longest = len(curr)
#                 curr = '' + l
#             else:
#                 curr += l
#         if len(curr) > longest:
#             return len(curr)
#         return longest

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        subs = ''
        longest = 0
        curr = ''
        for pointer in range(len(s)):
            subs = s[pointer] # string begin
            for c in range(pointer+1, len(s)):
                curr = s[c]
                if curr in subs:
                    if len(subs) > longest:
                        longest = len(subs)
                    break
                else:
                    subs += curr

            if len(subs) > longest:
                return len(subs)
        return longest


if __name__ == '__main__':
    s = Solution()
    res = s.lengthOfLongestSubstring("jbpnbwwd")
    print(res)
