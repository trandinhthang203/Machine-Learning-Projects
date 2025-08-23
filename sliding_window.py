class Sliding_window:
    def fixed_size(self, arr: list, k: int, t: int) -> int:
        if len(arr) < k:
            return None
        
        window_sum = sum(arr[:k])
        count = 0
        mean_sum = window_sum / k
        if mean_sum >= t:
            count += 1

        for i in range(k, len(arr)):
            window_sum += arr[i] - arr[i-k]
            mean_sum = window_sum / k
            if mean_sum >= t:
                count += 1

        return count

    def variable_size(self, arr: list, s: int) -> int:
        min_len = float('inf')
        window_sum = 0
        left = 0

        for right in range(len(arr)):
            window_sum += arr[right]
            while window_sum >= s:
                min_len = min(min_len, right - left + 1)
                window_sum -= arr[left]
                left += 1

        return 0 if min_len == float('inf') else min_len

    def longest_substring(self, s: str) -> int:
        seen = set()
        left = 0
        max_len = 0

        for right in range(len(s)):
            while s[right] in seen:   # nếu trùng thì bỏ bớt bên trái
                seen.remove(s[left])
                left += 1
            seen.add(s[right])
            max_len = max(max_len, right - left + 1)

        return max_len

    def maximumUniqueSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        seen = set()

        for right in range(len(nums)):
            while nums[right] in seen:
                seen.remove(nums[left])
                left += 1

            print(seen)
            seen.add(nums[right])

        return sum(seen)
    

str = [2,3,1,2,4,3]
# arr = [2,3,1,2,4,3]
# arr_set = set(arr)

# print(len(set(s)))
# print(len(arr_set))
# k = 7
# T = 3
s = Sliding_window()
print(s.maximumUniqueSubarray(str))

