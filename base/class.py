class Employee:
    ## Hàm gọi mặc định khi goi class
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

    def get_full_name(self):
        return f'Name:{self.name}, email:{self.email}'

    @classmethod ## nhận tham số mặc định là class chính nó
    def get_employ(cls):
        return cls('thang', 19, 'khuog')
    
    @staticmethod ## không nhận self hay cls
    def sum(a, b):
        return a + b


## Kết thừa

class Saler(Employee):
    def __init__(self, name, age, email, product):
        super().__init__(name, age, email)
        self.product = product

e = Employee('Trần Đình Thắng', 22, 'tranthangkhuong203@gmai.com')

print(e.get_full_name())
print(Employee.sum(5, 10))
