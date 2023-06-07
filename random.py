import random

nums = ['邱伟涵', '江卓能', '梁晓嘉', '汪茜茜', '江昭', '林志', '苗凌风', '祝小东', '袁晓锋']
menu = []
while len(menu) < len(nums):
    num = random.choice(nums)
    if num not in menu:
        menu.append(num)

print(menu)
