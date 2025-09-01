class Solution(object):
    def isValid(self, s):
        stack = []
        mapping = {'(':')', '[':']', '{':'}'}

        for i in s:
            if i in mapping:  
                stack.append(i)
            else:  
                if not stack or mapping[stack[-1]] != i:
                    return False
                stack.pop()

        return not stack

    
str = list('([)')
s = Solution()
print(s.isValid(str))