<<<<<<< HEAD
import numpy as np
# 从模型中验证过程得到的混淆矩阵为行之和为真实值，         预测值
#                                     即，  真实值
# 定义混淆矩阵
confusion_matrix = np.array([[832, 41, 22],
                             [41, 76, 12],
                             [29, 8, 36]])

# 计算每个类别的 TP，FN 和 FP
TP = np.diag(confusion_matrix)
FN = np.sum(confusion_matrix, axis=1) - TP#[13,22,31]
FP = np.sum(confusion_matrix, axis=0) - TP#[18,22,26]

# 计算精度、精确度、召回率和 F1 分数
accuracy = np.sum(TP) / np.sum(confusion_matrix)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

# 打印每个类别的度量值，并将结果写入一个文档中
with open("metrics.txt", "w") as file:
    for i in range(len(TP)):
        file.write(f"Class {i+1}:\n")
        file.write(f"TP: {TP}\n")
        file.write(f"FN: {FN:}\n")
        file.write(f"FP: {FP:}\n")
        file.write(f"Accuracy: {accuracy:.3f}\n")
        file.write(f"Precision: {precision[i]:.3f}\n")
        file.write(f"Recall: {recall[i]:.3f}\n")
        file.write(f"F1-score: {f1_score[i]:.3f}\n\n")
        print(f"Class {i+1}:")
        print(f"TP: {TP}\n")
        print(f"FN: {FN:}\n")
        print(f"FP: {FP:}\n")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision[i]:.3f}")
        print(f"Recall: {recall[i]:.3f}")
        print(f"F1-score: {f1_score[i]:.3f}")
        print()

=======
import numpy as np
# 从模型中验证过程得到的混淆矩阵为行之和为真实值，         预测值
#                                     即，  真实值
# 定义混淆矩阵
confusion_matrix = np.array([[832, 41, 22],
                             [41, 76, 12],
                             [29, 8, 36]])

# 计算每个类别的 TP，FN 和 FP
TP = np.diag(confusion_matrix)
FN = np.sum(confusion_matrix, axis=1) - TP#[13,22,31]
FP = np.sum(confusion_matrix, axis=0) - TP#[18,22,26]

# 计算精度、精确度、召回率和 F1 分数
accuracy = np.sum(TP) / np.sum(confusion_matrix)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

# 打印每个类别的度量值，并将结果写入一个文档中
with open("metrics.txt", "w") as file:
    for i in range(len(TP)):
        file.write(f"Class {i+1}:\n")
        file.write(f"TP: {TP}\n")
        file.write(f"FN: {FN:}\n")
        file.write(f"FP: {FP:}\n")
        file.write(f"Accuracy: {accuracy:.3f}\n")
        file.write(f"Precision: {precision[i]:.3f}\n")
        file.write(f"Recall: {recall[i]:.3f}\n")
        file.write(f"F1-score: {f1_score[i]:.3f}\n\n")
        print(f"Class {i+1}:")
        print(f"TP: {TP}\n")
        print(f"FN: {FN:}\n")
        print(f"FP: {FP:}\n")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision[i]:.3f}")
        print(f"Recall: {recall[i]:.3f}")
        print(f"F1-score: {f1_score[i]:.3f}")
        print()

>>>>>>> 90df61a1eb4949d38c3583627ea6ba339ab9ea7a
