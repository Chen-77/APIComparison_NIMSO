<<<<<<< HEAD
# # 将实体识别出来，并将实体写入到名为“final.txt”的文件中
# # 将测试集与预测出的标签分别读到a和b中
# 第一段代码只是将实体识别出来写入到新文件，而未与原文件中内容进行对应
# with open("./aspect_data/Simple_dataset/entitydata.txt", encoding="utf-8") as f1:
#     a = []
#     b = []
#     with open("./ner_data/Entitypaper_data/Ner_result_covert/test_prediction.txt", encoding="utf-8") as f:
#         for line in f1:
#             a.append(line.split(" "))
#         for line in f:
#             b.append(line.split(" "))
#
# # 根据[SEP]对每一行的实体进行识别，并存入all中
# all = []
# for i in range(0, len(a)):
#     tmp = []
#     string = ""
#     for j in range(0, len(a[i])):
#         # print(j)
#         if b[i][j + 1] != "O" and b[i][j + 2] == "[SEP]\n":
#             string = string + a[i][j]
#             tmp.append(string)
#             # all.append(tmp)
#             string = ""
#             break
#         elif b[i][j + 1] != "O" and b[i][j + 2] != "O":
#             string = string + a[i][j]
#         elif b[i][j + 1] == "O" and b[i][j + 2] == "[SEP]\n":
#             break
#
#         elif b[i][j + 1] != "O" and b[i][j + 2] == "O":
#             string = string + a[i][j]
#             tmp.append(string)
#
#             string = ""
#
#     aa = set(tmp)
#     tmp = list(aa)
#     all.append(tmp)
#
# 将识别的实体写入final.txt中
# with open("./aspect_data/Simple_dataset/newfinal.txt", "w", encoding="utf-8") as f:
#     for i in all:
#         if len(i) == 0:
#             f.write("\n")
#             continue
#         for j in range(0, len(i) - 1):
#             # f.write(i[j].strip+' ')
#             string = ""
#             for k in i[j].strip('",?. '):
#                 string = string + k
#             f.write(string + ' ')
#         f.write(i[len(i) - 1].replace('"', '').replace('?', '').replace(',', '').replace(' ', '').replace('!','').replace(':','').strip('",?. ') + "\n")
# with open("./aspect_data/Simple_dataset/Top1-5912_Simple_entitydata.txt", encoding="utf-8") as f1, \
#         open("./ner_data/Entitypaper_data/Ner_result_covert/test_prediction.txt", encoding="utf-8") as f, \
#         open("./aspect_data/Simple_dataset/newfinal.txt", "w", encoding="utf-8") as fout:
#     for line1, line2 in zip(f1, f):
#         a = line1.split(" ")
#         b = line2.split(" ")
#
#         # 根据[SEP]对每一行的实体进行识别，并存入all中
#         tmp = []
#         string = ""
#         for j in range(0, len(a)):
#             if b[j + 1] != "O" and b[j + 2] == "[SEP]\n":
#                 string = string + a[j]
#                 tmp.append(string)
#                 string = ""
#                 break
#             elif b[j + 1] != "O" and b[j + 2] != "O":
#                 string = string + a[j]
#             elif b[j + 1] == "O" and b[j + 2] == "[SEP]\n":
#                 break
#             elif b[j + 1] != "O" and b[j + 2] == "O":
#                 string = string + a[j]
#                 tmp.append(string)
#                 string = ""
#
#         aa = set(tmp)
#         tmp = list(aa)
#
#         # 将识别的实体写入newfinal.txt中
#         fout.write(line1.strip())
#         for i in tmp:
#             string = ""
#             for k in i.strip('",?. '):
#                 string = string + k
#             fout.write('\t' + string.replace('"', '').replace('?', '').replace(',', '').replace(' ', '').replace('!','').replace(':', '').strip('",?. '))
#         fout.write('\n')
with open("./aspect_data/Rel_dataset/Top1-5912_allsenpair.tsv", encoding="utf-8") as f1, \
     open("./aspect_data/Rel_dataset/test_prediction.txt", encoding="utf-8") as f2, \
     open("./aspect_data/Rel_dataset/newfinal.txt", "w", encoding="utf-8") as f3:

    for line1, line2 in zip(f1, f2):
        entities = []
        words = line1.strip().split(" ")
        labels = line2.strip().split(" ")
        current_entity = ""
        for word, label in zip(words, labels[1:]):
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = word
            elif label.startswith("I-"):
                current_entity += " " + word
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = ""
        if current_entity:
            entities.append(current_entity)

        if len(entities) == 0:
            f3.write(line1.strip() + "\n")
        else:
            for entity in entities:
                f3.write(line1.strip() + "\t" + entity.replace('"', '').replace('?', '').replace(',', '').replace(' ', '').replace('!', '').replace(':', '').replace(';', '').strip('.') + "\n")
=======
# # 将实体识别出来，并将实体写入到名为“final.txt”的文件中
# # 将测试集与预测出的标签分别读到a和b中
# 第一段代码只是将实体识别出来写入到新文件，而未与原文件中内容进行对应
# with open("./aspect_data/Simple_dataset/entitydata.txt", encoding="utf-8") as f1:
#     a = []
#     b = []
#     with open("./ner_data/Entitypaper_data/Ner_result_covert/test_prediction.txt", encoding="utf-8") as f:
#         for line in f1:
#             a.append(line.split(" "))
#         for line in f:
#             b.append(line.split(" "))
#
# # 根据[SEP]对每一行的实体进行识别，并存入all中
# all = []
# for i in range(0, len(a)):
#     tmp = []
#     string = ""
#     for j in range(0, len(a[i])):
#         # print(j)
#         if b[i][j + 1] != "O" and b[i][j + 2] == "[SEP]\n":
#             string = string + a[i][j]
#             tmp.append(string)
#             # all.append(tmp)
#             string = ""
#             break
#         elif b[i][j + 1] != "O" and b[i][j + 2] != "O":
#             string = string + a[i][j]
#         elif b[i][j + 1] == "O" and b[i][j + 2] == "[SEP]\n":
#             break
#
#         elif b[i][j + 1] != "O" and b[i][j + 2] == "O":
#             string = string + a[i][j]
#             tmp.append(string)
#
#             string = ""
#
#     aa = set(tmp)
#     tmp = list(aa)
#     all.append(tmp)
#
# 将识别的实体写入final.txt中
# with open("./aspect_data/Simple_dataset/newfinal.txt", "w", encoding="utf-8") as f:
#     for i in all:
#         if len(i) == 0:
#             f.write("\n")
#             continue
#         for j in range(0, len(i) - 1):
#             # f.write(i[j].strip+' ')
#             string = ""
#             for k in i[j].strip('",?. '):
#                 string = string + k
#             f.write(string + ' ')
#         f.write(i[len(i) - 1].replace('"', '').replace('?', '').replace(',', '').replace(' ', '').replace('!','').replace(':','').strip('",?. ') + "\n")
# with open("./aspect_data/Simple_dataset/Top1-5912_Simple_entitydata.txt", encoding="utf-8") as f1, \
#         open("./ner_data/Entitypaper_data/Ner_result_covert/test_prediction.txt", encoding="utf-8") as f, \
#         open("./aspect_data/Simple_dataset/newfinal.txt", "w", encoding="utf-8") as fout:
#     for line1, line2 in zip(f1, f):
#         a = line1.split(" ")
#         b = line2.split(" ")
#
#         # 根据[SEP]对每一行的实体进行识别，并存入all中
#         tmp = []
#         string = ""
#         for j in range(0, len(a)):
#             if b[j + 1] != "O" and b[j + 2] == "[SEP]\n":
#                 string = string + a[j]
#                 tmp.append(string)
#                 string = ""
#                 break
#             elif b[j + 1] != "O" and b[j + 2] != "O":
#                 string = string + a[j]
#             elif b[j + 1] == "O" and b[j + 2] == "[SEP]\n":
#                 break
#             elif b[j + 1] != "O" and b[j + 2] == "O":
#                 string = string + a[j]
#                 tmp.append(string)
#                 string = ""
#
#         aa = set(tmp)
#         tmp = list(aa)
#
#         # 将识别的实体写入newfinal.txt中
#         fout.write(line1.strip())
#         for i in tmp:
#             string = ""
#             for k in i.strip('",?. '):
#                 string = string + k
#             fout.write('\t' + string.replace('"', '').replace('?', '').replace(',', '').replace(' ', '').replace('!','').replace(':', '').strip('",?. '))
#         fout.write('\n')
with open("./aspect_data/Rel_dataset/Top1-5912_allsenpair.tsv", encoding="utf-8") as f1, \
     open("./aspect_data/Rel_dataset/test_prediction.txt", encoding="utf-8") as f2, \
     open("./aspect_data/Rel_dataset/newfinal.txt", "w", encoding="utf-8") as f3:

    for line1, line2 in zip(f1, f2):
        entities = []
        words = line1.strip().split(" ")
        labels = line2.strip().split(" ")
        current_entity = ""
        for word, label in zip(words, labels[1:]):
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = word
            elif label.startswith("I-"):
                current_entity += " " + word
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = ""
        if current_entity:
            entities.append(current_entity)

        if len(entities) == 0:
            f3.write(line1.strip() + "\n")
        else:
            for entity in entities:
                f3.write(line1.strip() + "\t" + entity.replace('"', '').replace('?', '').replace(',', '').replace(' ', '').replace('!', '').replace(':', '').replace(';', '').strip('.') + "\n")
>>>>>>> 90df61a1eb4949d38c3583627ea6ba339ab9ea7a
