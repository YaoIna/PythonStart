vowels = ['a', 'e', 'i', 'o', 'u']
word = "Milliways"
found = []
for i in word:
    if i not in found:
        found.append(i)
        print(i)