class hash_table:
    def romanToInt(self, s: str) -> int:
        roman = {
            'I': 1, 'V': 5, 'X': 10,
            'L': 50, 'C': 100, 'D': 500, 'M': 1000
        }
        
        total = 0
        for i in range(len(s) - 1):
            if roman[s[i]] < roman[s[i + 1]]:
                total -= roman[s[i]]
            else:
                total += roman[s[i]]
        
        total += roman[s[-1]]
        return total
    
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        roman_map = {
            1000: "M", 900: "CM", 500: "D", 400: "CD", 
            100: "C", 90: "XC", 50: "L", 40: "XL", 
            10: "X", 9: "IX", 5: "V", 4: "IV", 1: "I"
        }
        key = list(roman_map.keys())
        result = ''

        for i in key:
            while num >= i:
                num -= i
                result += roman_map[i]
                print(result)

        return result
    
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        : using backtracking
        """
        combinations = {
            '2' : 'abc',
            '3' : 'def',
            '4' : 'ghi',
            '5' : 'jkl',
            '6' : 'mno',
            '7' : 'pqrs',
            '8' : 'tuv',
            '9' : 'wxyz'
        }
        for i in range(len(digits)):
            print(combinations[digits[i]])

    def isValidSudoku(self, board: list[list[str]]):
        """
        :type board: List[List[str]]
        :rtype: bool
        """

        for i in range(9):
            temp = set()
            for j in range(9):
                if board[i][j] in temp:
                    return False
                elif board[i][j] != '.':
                    temp.add(board[i][j])

        for i in range(9):
            temp = set()
            for j in range(9):
                if board[j][i] in temp:
                    return False
                elif board[j][i] != '.':
                    temp.add(board[j][i])

        start = [
            (0, 0), (0, 3), (0, 6),
            (3, 0), (3, 3), (3, 6),
            (6, 0), (6, 3), (6, 6),
        ]
        for r, c in start:
            temp = set()
            for i in range(r, r+3):
                for j in range(c, c+3):
                    if board[i][j] in temp:
                        return False
                    elif board[i][j] != '.':
                        temp.add(board[i][j])

        return True

        
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums_positive = [x for x in nums if x > 0]  # dùng set thay list
        nums_positive.sort()
        if nums_positive:
            
            if min(nums_positive) > 1:
                return 1
            
            if len(set(nums_positive)) == (nums_positive[-1] - nums_positive[0] + 1):
                return nums_positive[-1] + 1
            
            for i in nums_positive:
                if i + 1 not in nums_positive:
                    return i + 1
                           
        else:
            return 1

    def groupAnagrams(self, strs):
        groups = {}
        for word in strs:
            key = ''.join(sorted(word))  
            if key not in groups:
                groups[key] = []
            groups[key].append(word)
        return list(groups.values())

strs = ["eat","tea","tan","ate","nat","bat"]
s = ''.join(sorted(strs[0]))
print(s)
h = hash_table()
print(h.groupAnagrams(strs))