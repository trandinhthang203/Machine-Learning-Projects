import re

class Solution(object):
    def romanToInt(self, s = 'MCMXCIV'):
        """
        :type s: str
        :rtype: int
        """
        list = [['I', 1], ['V', 5], ['X', 10], ['L', 50], ['C', 100], ['D', 500], ['M', 1000]]
        for i in s:
            if i in list[:]:
                pass
        # sum = 0

        # for index, value in enumerate(list):
        #     for index2, value2 in enumerate(list, start=index+1):
        #         if index2 <= len(list):
        #             if list[index][1] < list[index2][1]:
        #                 sum = list[index2][1] - list[index][1]
        #             else:
        #                 sum += list[index][1]
        # return sum

# Input: s = "MCMXCIV"
# Output: 1994
# Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.

s = Solution()
s.romanToInt()