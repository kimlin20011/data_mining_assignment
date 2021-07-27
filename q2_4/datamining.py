import pandas as pd
import numpy as np

file = "uriage6.csv"

df = pd.read_csv(file, names=['customers', '1', '2', '3',
                              '4', '5', '6', '7', '8', '9', '10'], encoding='UTF-8')
# print(df)

category_list = ['たばこ', 'パン類', '乳製品', '肉', '調味料', '酒類', '野菜', '雑貨', '飲料', '魚']
category_map = {
    'たばこ': 0,
    'パン類': 1,
    '乳製品': 2,
    '肉': 3,
    '調味料': 4,
    '酒類': 5,
    '野菜': 6,
    '雑貨': 7,
    '飲料': 8,
    '魚': 9
}
category_name = list(category_map.keys())
print(f'category name: {category_name}')
# print(category_name[0])

customer_len = 10000

np_arr = np.zeros((customer_len, len(category_list)), dtype=bool)

for i in range(1, 11):
    column_index = str(i)
    column = df[column_index]
    for j in range(customer_len):
        if isinstance(column[j], str):
            col = category_map[column[j]]
            np_arr[j, col] = True

# print(np_arr[:, 2])
# print(np_arr[:, 2].sum())
# p = 2
# q = 3
# print((np_arr[:, p] & np_arr[:, q]).sum())
# print((np_arr[:, p] | np_arr[:, q]).sum())

for p_i in range(10):
    for q_i in range(10):
        if p_i == q_i:
            continue

        a = (np_arr[:, p_i] & np_arr[:, q_i]).sum()
        b = np_arr[:, p_i].sum()

        acc = round(a / b, 2)
        # print(f'acc: {acc}')
        if a / b > 0.6 and b > 1000:
            print(
                f'p: {category_name[p_i]}, q: {category_name[q_i]}, acc: {acc}')


# df = pd.read_csv(file, names=['customers','1','2','3','4','5','6','7','8','9'], encoding='Shift-JIS').fillna('').set_index('customers')
# print(df)

# customers_record = df.apply(pd.Series.value_counts, axis=1).fillna(0).replace(1, 1)
# cr = pd.DataFrame(customers_record)
# print(cr)

# num_customer = 10000


# categoryList = []
# for i in cr.columns:
#     categoryList.append(i)
# print(categoryList)

# ele = cr[categoryList[1]][0]
# print(ele)
# print(type(ele))

# a = sum( (cr.たばこ == '1')  & (cr.パン類 == '1') & (cr.野菜 == '1'))
# b = sum( (cr.たばこ == '1')  & (cr.パン類 == '1'))

# print('a = ',a)
# print('b = ',b)

# category_sum = customers_record[customers_record=='1'].count(0)
# print(df)
# print(customers_record)
# print(category_sum)
