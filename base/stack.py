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
                else:
                    stack.pop()

        return not stack

    def simplifyPath(self, path: str) -> str:
        pass
    
str = list('([)')
s = Solution()
print(s.simplifyPath(str))