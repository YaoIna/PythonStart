vowels = {"a", "e", "o", "i", "u"}
word_set = set(input("Provide a word for searching vowels:"))
word_vowels = vowels.intersection(word_set)
for i in sorted(word_vowels):
    print(i)
