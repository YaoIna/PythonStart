
def find_cody_words_number(n: int) -> int:
    if n <= 0:
        return 0

    x = list()
    y = list()
    z = list()
    x.append(2)
    y.append(3)
    z.append(x[0] + y[0])
    for i in range(1, n):
        x.append(2 * y[i - 1])
        y.append(3 * x[i - 1] + 2 * y[i - 1])
        z.append(x[i] + y[i])

    return z[n - 1]


number = int(input("input number of letters:"))
print("The number of cody words of", number, "letters is", find_cody_words_number(number))
