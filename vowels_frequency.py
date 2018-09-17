vowels_found = {"a": 0, "e": 0, "i": 0, "o": 0, "u": 0}
word = input("Provide a word for searching for vowels:")
for i in word:
    if i in vowels_found:
        vowels_found[i] += 1
for k, v in sorted(vowels_found.items()):
    if v == 0:
        continue
    time_string = "times"
    if v <= 1:
        time_string = "time"
    print(k, "was found", v, time_string)
