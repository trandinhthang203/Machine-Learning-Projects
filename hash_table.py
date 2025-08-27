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

h = hash_table()
print(h.letterCombinations('23'))