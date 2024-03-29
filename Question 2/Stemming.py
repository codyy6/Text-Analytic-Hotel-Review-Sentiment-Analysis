import re
from nltk.stem import PorterStemmer, LancasterStemmer
file = open ("Question 2\Data_1.txt", encoding='utf8')
words = file.read()

words = re.findall(r'\b\w+\b',words.lower())

regex_stemmer = lambda w: re.sub(r's$|es$|ed$|ing$','',w)

porter = PorterStemmer()
porter_stems = [porter.stem(word) for word in words]

lancaster = LancasterStemmer()
lancaster_stems = [lancaster.stem(word) for word in words]

print ("Word\t\tRegular Expression\t\tPorter Stemmer\t\tLancaster Stemmer")
print("-"*70)
for i in range(len(words)):
    print(f"{words[i]:<15}\t\t{regex_stemmer(words[i]):<22}\t\t{porter_stems[i]:<15}\t\t{lancaster_stems[i]:<18}")

with open("stemming_output.txt", "w", encoding="utf8") as output_file:
    output_file.write("Word\t\tRegular Expression\t\tPorter Stemmer\t\tLancaster Stemmer\n")
    output_file.write("-"*70)
    for i in range(len(words)):
        output_file.write(f"{words[i]:<15}\t\t{regex_stemmer(words[i]):<22}\t\t{porter_stems[i]:<15}\t\t{lancaster_stems[i]:<18}\n")