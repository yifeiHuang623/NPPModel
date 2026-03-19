import torch

def generate_tensor_of_distribution(time):
    list1=[]
    temp=[i for i in range(time)]
    for i in range(time):
        if i == time//2:
            list1.append(temp)
        elif i<time//2:
            list1.append(temp[-(time//2-i):]+temp[:-(time//2-i)])
        else :
            list1.append(temp[(i-time//2):]+temp[:(i-time//2)])
    return torch.tensor(list1)