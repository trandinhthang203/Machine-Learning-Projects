# comprehention là cách viết ngắn gọn hiệu quả để tạo ra các cấu trúc dữ liệu như list, set, dict

list_key = [1, 4, 2, 6, 8]
list_copy = list_key.copy()
print(list_copy)
list_name = ['thang', 'quang', 'thanh', 'khuong']

# list comprehention
list_com = [x*x for x in list_key if x > 5]
list_com = [x*x if x < 6 else 'big than' for x in list_key]

print(list(zip(list_key, list_name)))

dict_demo = {}
for fn, ln in zip(list_key, list_name):
    dict_demo[ln] = fn

print(dict_demo)

# dict comprehention
dict_comp = {last_name : first_name for first_name, last_name in zip(list_key, list_name) if last_name == 'thang'}
print(dict_comp)


# set comprehention (dict sẽ trả về 1 cặp key-value and set trả về 1 giá trị thuii)
set_comp = {i for i in list_key}
print(set_comp)