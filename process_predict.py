
with open("./ner_data/test.txt", encoding="utf8") as f1:
    a = []
    b = []
    with open("./output/test_prediction.txt") as f:
        for line in f1:
            a.append(line.split(" "))
        for line in f:
            b.append(line.split(" "))

all = []
for i in range(0, len(a)):
    tmp = []
    string = ""
    for j in range(0, len(a[i])):
        print(j)
        if b[i][j + 1] != "O" and b[i][j + 2] == "[SEP]\n":
            string = string + a[i][j]
            tmp.append(string)
            # all.append(tmp)
            string = ""
            break
        elif b[i][j + 1] != "O" and b[i][j + 2] != "O":
            string = string + a[i][j]
        elif b[i][j + 1] == "O" and b[i][j + 2] == "[SEP]\n":
            break

        elif b[i][j + 1] != "O" and b[i][j + 2] == "O":
            string = string + a[i][j]
            tmp.append(string)

            string = ""

    aa = set(tmp)
    tmp = list(aa)
    all.append(tmp)

with open("final.txt", "w", encoding="utf8") as f:
    for i in all:
        if len(i) == 0:
            f.write("\n")
            continue
        for j in range(0, len(i) - 1):
            # f.write(i[j].strip+' ')
            string = ""
            for k in i[j].strip():
                string = string + k
            f.write(string + ' ')
        f.write(i[len(i) - 1].strip() + "\n")
