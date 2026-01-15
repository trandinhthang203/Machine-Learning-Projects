## List trong Python - có thể thay đổi, có thể chứa các kiểu dữ liệu khác nhau
## các method thường dùng

# append - thêm một phần tử vào cuối danh sách
l  = [5, 2, 4, 1, 3]
l.append('thang')

# list_copy - tạo một bản sao của danh sách
l_copy = l.copy()


list_new = l
list_new[1] = 100  # thay đổi giá trị của phần tử đầu tiên trong list_new
# list_new và l đều tham chiếu đến cùng một đối tượng trong bộ nhớ, nên thay đổi trong list_new cũng sẽ ảnh hưởng đến l

# extend - thêm các phần tử từ một iterable vào cuối danh sách
l.extend([6, 7, 8])

# insert - chèn một phần tử vào vị trí chỉ định
l.insert(2, ['new', 'element'])

# remove - xóa phần tử đầu tiên có giá trị nhất định, nếu không tìm thấy sẽ báo lỗi
l.remove('thang')

# pop - xóa và trả về phần tử ở vị trí chỉ định, nếu không chỉ định thì sẽ xóa phần tử cuối cùng
l.pop(2)

# clear - xóa tất cả các phần tử trong danh sách
# l.clear()

# sort - sắp xếp danh sách theo thứ tự tăng dần, nếu muốn sắp xếp giảm dần thì thêm tham số reverse=True
# sort với key - sắp xếp theo một tiêu chí nhất định
# Ví dụ: sắp xếp theo phần tử thứ hai của tuple trong danh sách
l.sort(reverse=True)
# l_sorted = l.sort() => None

# reverse - đảo ngược thứ tự các phần tử trong danh sách
# l.reverse()

# sorted - trả về một danh sách mới đã được sắp xếp, không thay đổi danh sách gốc
l_sorted = sorted(l)

# max, min - trả về phần tử lớn nhất, nhỏ nhất trong danh sách
max_value = min(l)

# sum - trả về tổng các phần tử trong danh sách
total_sum = sum(l) 

# index - trả về chỉ số của phần tử đầu tiên có giá trị nhất định, nếu không tìm thấy sẽ báo lỗi
index_of_4 = l.index(1)

# enumerate - trả về một iterator của các tuple (index, value) cho mỗi phần tử trong danh sách
for index, value in enumerate(l, start=3): # bắt đầu từ chỉ số 3
    print(f"Index: {index}, Value: {value}")


# split - chia một chuỗi thành danh sách các chuỗi con dựa trên ký tự phân tách
s = 'a   ,  b,c,d,          e'
l_from_string = s.split('b')

## tuple -  không thể thay đổi, không có các method như append, extend, insert, remove, pop, clear, sort, reverse
tuple_example = ([10, 20], 2, 3, 4, 5)
# tuple_example[0] = 10  # sẽ báo lỗi vì tuple không thể thay đổi
# nếu phân tử là một danh sách thì có thể thay đổi nội dung của danh sách đó
tuple_example[0][0] = 120  # sẽ không báo lỗi vì chúng ta đang thay đổi nội dung của danh sách trong tuple

list_example = list(tuple_example) # chuyển đổi tuple thành list
list_example[0] = [100, 200]  # sẽ không báo lỗi vì chúng ta đang thay đổi nội dung của list

## set - tập hợp, không có thứ tự, không có phần tử trùng lặp, không thể thay đổi nhưng có thể thêm hoặc xóa phần tử
# set không có các method như append, extend, insert, remove, pop, clear,...
# set không có thứ tự, không thể truy cập phần tử bằng chỉ số, nhưng có thể kiểm tra sự tồn tại của phần tử
#  có thể thay đổi nội dung của các phần tử bên trong nếu chúng là kiểu dữ liệu có thể thay đổi (như list)
set_example = {1, 2, 3, 4, 5}

set_example.add('tranthangkhuong203@gmail.com')  # thêm phần tử vào tập hợp
set_example.remove(1)  # xóa phần tử khỏi tập hợp, nếu không tìm thấy sẽ báo lỗi

# intersection - trả về tập hợp các phần tử chung giữa hai tập hợp
set_a = {1, 2, 3}
set_b = {3, 4, 5}
set_c = {5, 6, 7}
intersection_set = set_a.intersection(set_b, set_c)  # {}

# difference - trả về tập hợp các phần tử chỉ có trong tập hợp đầu tiên
difference_set = set_a.difference(set_b)  # {1, 2}

# union - trả về tập hợp các phần tử có trong ít nhất một trong các tập hợp
union_set = set_a.union(set_b, set_c)  # {1, 2, 3, 4, 5, 6, 7}

print(union_set)  


## dictionary - từ điển, là một tập hợp các cặp khóa-giá trị, có thể thay đổi
# dictionary không có thứ tự, không có phần tử trùng lặp, không thể truy cập phần tử bằng chỉ số
# có thể thêm, xóa, sửa đổi các cặp khóa-giá trị
# khóa phải là kiểu dữ liệu bất biến (immutable) như int, str, tuple, không thể là list hay set
# khóa là giá trị duy nhất trong từ điển, không thể có hai khóa giống nhau, nếu có hai khóa giống nhau thì khóa sau sẽ ghi đè lên khóa trước

dict_example = {
    'name': 'Tran Thang Khuong',
    'age': 30,
    'email': 'tranthangkhuong203@gmail.com'
}

# truy cập giá trị theo khóa
print(dict_example['name'])  # 'Tran Thang Khuong'

# xóa cặp khóa-giá trị
del dict_example['age']  # xóa cặp khóa-giá trị có khóa 'age'

# thêm cặp khóa-giá trị mới
dict_example['address'] = '123 Street Name'

print(dict_example.get('emaili', 'deo co'))  # 

# update - cập nhật giá trị của một khóa, nếu khóa không tồn tại thì sẽ thêm cặp khóa-giá trị mới
dict_example.update({'email': 'tahngsng', 'phone': '1234567890'})

# pop - xóa và trả về giá trị của một khóa, nếu khóa không tồn tại thì sẽ báo lỗi
email_value = dict_example.pop('gmail', 'deo co')  # nếu không có khóa 'email' thì sẽ trả về 'deo co'


# keys, values, items - trả về các đối tượng view của các khóa, giá trị, cặp khóa-giá trị
keys_view = dict_example.keys()  # trả về các khóa
values_view = dict_example.values()  # trả về các giá trị
items_view = dict_example.items()  # trả về các cặp khóa-giá trị


class Solution(object):
    def is_duplicate(self, s):
        """
        :type s: str
        :rtype: bool
        """
        seen = set()
        for char in s:
            if char in seen:
                return True
            seen.add(char)
        return False
    
    def lengthOfLongestSubstring(self, text):
        max_lenght = 0
        result = []
        for index, sub_text_1 in enumerate(text):
            for index_2, sub_text_2 in enumerate(text, start=index + 1):
                if index_2 <= len(text) and self.is_duplicate(text[index:index_2]) == False:
                    if len(text[index:index_2]) > max_lenght:
                        max_lenght = len(text[index:index_2])
                        result = text[index:index_2]

        return len(result)

text = "abcabcbb"
s = Solution()
print(s.lengthOfLongestSubstring(text))