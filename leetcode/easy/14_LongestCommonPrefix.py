class Solution(object):
    def longestCommonPrefix(self, strs) -> str:
        """find the longest common prefix strong
        if no common prefix, return empty stirng ""

        Args:
            strs (_type_): List[str]
        """
        min_len = min([len(str) for str in strs])
        if min_len == 0:
            return ""
        index = 0
        result = ""
        while index < min_len:
            set_ex = set()
            for str in strs:
                print(str)
                set_ex.add(str[index])
            if len(set_ex) > 1:
                return result
            result += strs[0][index]
            index += 1
        return result
s = Solution()
strs = ["a"]
r = s.longestCommonPrefix(strs)
print(r)
