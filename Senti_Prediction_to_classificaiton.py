<<<<<<< HEAD
import os
import pandas as pd

if __name__ == '__main__':
    path = "output/Sentiment_classification/2023.4.27WORel_Single_output/new_WO_Mul"
    pd_all = pd.read_csv(os.path.join(path, "test_results.tsv"), sep='\t', header=None)

    data = pd.DataFrame(columns=['polarity'])
    print(pd_all.shape)

    for index in pd_all.index:
        neutral_score = pd_all.loc[index].values[0]
        positive_score = pd_all.loc[index].values[1]
        negative_score = pd_all.loc[index].values[2]

        if max(neutral_score, positive_score, negative_score) == neutral_score:
            # data.append(pd.DataFrame([index, "neutral"],columns=['id','polarity']),ignore_index=True)
            data.loc[index + 1] = ["o"]
        elif max(neutral_score, positive_score, negative_score) == positive_score:
            # data.append(pd.DataFrame([index, "positive"],columns=['id','polarity']),ignore_index=True)
            data.loc[index + 1] = ["p"]
        else:
            # data.append(pd.DataFrame([index, "negative"],columns=['id','polarity']),ignore_index=True)
            data.loc[index + 1] = ["n"]
        # print(negative_score, positive_score, negative_score)

    data.to_csv(os.path.join(path, "Rel_Mul_Senti_sample.tsv"), sep='\t')
=======
import os
import pandas as pd

if __name__ == '__main__':
    path = "output/Sentiment_classification/2023.4.27WORel_Single_output/new_WO_Mul"
    pd_all = pd.read_csv(os.path.join(path, "test_results.tsv"), sep='\t', header=None)

    data = pd.DataFrame(columns=['polarity'])
    print(pd_all.shape)

    for index in pd_all.index:
        neutral_score = pd_all.loc[index].values[0]
        positive_score = pd_all.loc[index].values[1]
        negative_score = pd_all.loc[index].values[2]

        if max(neutral_score, positive_score, negative_score) == neutral_score:
            # data.append(pd.DataFrame([index, "neutral"],columns=['id','polarity']),ignore_index=True)
            data.loc[index + 1] = ["o"]
        elif max(neutral_score, positive_score, negative_score) == positive_score:
            # data.append(pd.DataFrame([index, "positive"],columns=['id','polarity']),ignore_index=True)
            data.loc[index + 1] = ["p"]
        else:
            # data.append(pd.DataFrame([index, "negative"],columns=['id','polarity']),ignore_index=True)
            data.loc[index + 1] = ["n"]
        # print(negative_score, positive_score, negative_score)

    data.to_csv(os.path.join(path, "Rel_Mul_Senti_sample.tsv"), sep='\t')
>>>>>>> 90df61a1eb4949d38c3583627ea6ba339ab9ea7a
    # print(data)