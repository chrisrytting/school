def sub(a,b):
    diff_in_length = len(a) - len(b)
    b = [0]*diff_in_length + b
    total = [0]*len(a)
    i = len(a) - 1
    n = len(a)
    while i >= 0:
        if a[i] > b[i]:
            total[i] = a[i] - b[i]
        if b[i] > a[i]:
            total[i] = a[i] + 10 - b[i]
            a[i-1] -= 1
        i -= 1
    return total

a = [1,2,3]
b = [2,4]
print sub(a, b)

