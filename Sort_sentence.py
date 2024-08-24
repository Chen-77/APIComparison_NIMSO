<<<<<<< HEAD
#将同一问题下的评论进行笛卡尔积形式组合,
#根据预测结果进行排序
import itertools

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import  dataframe_to_rows
# def read_dataset(file):
# def sim_rel(df):
#     dist = []
#     n = 0
#     df_new = pd.DataFrame(columns=('string1', 'string2'))
#     for ca in df["comments"]:
#         for cb in df["comments"]:
#             try:
#                 ca_index = df[df.comments == ca].index.tolist()[0]
#                 cb_index = df[df.comments == cb].index.tolist()[0]
#                 if ca != cb and ca_index < cb_index:
#                     df2 = pd.DataFrame({'string1': [ca], 'string2': [cb]})
#                     df_new = pd.concat([df_new, df2], ignore_index=True)
#                     # print(df_new)
#                 n = n + 1
#             except:
#                 print(f'第{n}条数据处理失败'.format(ca))
#     return df_new
def sim_rel(df):
    df_new = pd.DataFrame(columns=('string1', 'string2'))
    for i in range(len(df)):
        if i == 0:
            df2 = pd.DataFrame({'string1': [" "], 'string2': [df.iloc[i]['comments']], 'votes':[df.iloc[i]['votes']]})
            df_new = pd.concat([df_new, df2], ignore_index=True)
        else:
            for j in range(i):
                try:
                    df2 = pd.DataFrame({'string1': [df.iloc[j]['comments']], 'string2': [df.iloc[i]['comments']], 'votes':[df.iloc[i]['votes']]})
                    df_new = pd.concat([df_new, df2], ignore_index=True)
                except:
                    print(f'第{i}条数据处理失败')
    return df_new
def sim_rel_sort(df):
    if len(df) > 1:
        df.sort_values(by='Num1', inplace=True, ascending=False)
        df_new_sort = df.apply(lambda x:x.iloc[0])
        return df_new_sort
# 用于制作负例
def sim_rel_nagative_sort(df):
    if len(df) > 1:
        df.sort_values(by='Num1', inplace=True, ascending=False)
        df_new_sort = df.apply(lambda x:x.iloc[-1])
        return df_new_sort
# 提取首句与空集
def sim_rel_first(df):
    if len(df) == 1:
        # df_new_sort = df.apply(lambda x: x.iloc[0])
        return df
if __name__ == '__main__':
    # 将同一问题下的评论进行笛卡尔积形式组合
    # xlsx_file = r"H:\BERT-CRF-for-NER\Comment_data\Top1-5912_questions_so_comment.xlsx"
    # comment_sheet = pd.read_excel(xlsx_file)
    # comment_row = comment_sheet.loc[0].values
    # print("\n{0}".format(comment_row))
    #
    # comment_sheet = pd.DataFrame(comment_sheet)
    # comment_group = comment_sheet.groupby("question_header", sort=False)
    #
    # ratings = comment_group.apply(sim_rel)
    #
    # ratings_df = pd.DataFrame(ratings)
    # ratings_df.reset_index(inplace=True)
    # ratings_df.to_excel("Top1-4456_senpair.xlsx")

    # 根据预测结果进行排序
    xlsx_file_sort = r"H:\BERT-CRF-for-NER\Comment_data\Top1-4456_senpair_prediction.xlsx"
    comment_sheet_sort = pd.read_excel(xlsx_file_sort)
    # comment_row_sort = comment_sheet_sort.loc[0].values
    # print("\n{0}".format(comment_row_sort))
    # print("\n{0}".format(comment_sheet_sort))
    comment_sheet_sort = pd.DataFrame(comment_sheet_sort)
    # print("\n{0}".format(comment_sheet_sort))
    comment_group_sort = comment_sheet_sort.groupby("string2", sort=False, group_keys=False, as_index=False)
    ratings_sort = comment_group_sort.apply(sim_rel_sort)
    ratings_sort_na = comment_group_sort.apply(sim_rel_nagative_sort)
    ratings_sort_first = comment_group_sort.apply(sim_rel_first)
    ratings_df_sort = pd.DataFrame(ratings_sort)
    ratings_df_sort_na = pd.DataFrame(ratings_sort_na)
    ratings_df_sort_first = pd.DataFrame(ratings_sort_first)

    ratings_df_sort.reset_index(inplace=False)
    ratings_df_sort_na.reset_index(inplace=False)
    ratings_df_sort_first.reset_index(inplace=False)
    ratings_df_sort.to_excel("Top1-4456_senpair_sorted.xlsx")



=======
#将同一问题下的评论进行笛卡尔积形式组合,
#根据预测结果进行排序
import itertools

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import  dataframe_to_rows
# def read_dataset(file):
# def sim_rel(df):
#     dist = []
#     n = 0
#     df_new = pd.DataFrame(columns=('string1', 'string2'))
#     for ca in df["comments"]:
#         for cb in df["comments"]:
#             try:
#                 ca_index = df[df.comments == ca].index.tolist()[0]
#                 cb_index = df[df.comments == cb].index.tolist()[0]
#                 if ca != cb and ca_index < cb_index:
#                     df2 = pd.DataFrame({'string1': [ca], 'string2': [cb]})
#                     df_new = pd.concat([df_new, df2], ignore_index=True)
#                     # print(df_new)
#                 n = n + 1
#             except:
#                 print(f'第{n}条数据处理失败'.format(ca))
#     return df_new
def sim_rel(df):
    df_new = pd.DataFrame(columns=('string1', 'string2'))
    for i in range(len(df)):
        if i == 0:
            df2 = pd.DataFrame({'string1': [" "], 'string2': [df.iloc[i]['comments']], 'votes':[df.iloc[i]['votes']]})
            df_new = pd.concat([df_new, df2], ignore_index=True)
        else:
            for j in range(i):
                try:
                    df2 = pd.DataFrame({'string1': [df.iloc[j]['comments']], 'string2': [df.iloc[i]['comments']], 'votes':[df.iloc[i]['votes']]})
                    df_new = pd.concat([df_new, df2], ignore_index=True)
                except:
                    print(f'第{i}条数据处理失败')
    return df_new
def sim_rel_sort(df):
    if len(df) > 1:
        df.sort_values(by='Num1', inplace=True, ascending=False)
        df_new_sort = df.apply(lambda x:x.iloc[0])
        return df_new_sort
# 用于制作负例
def sim_rel_nagative_sort(df):
    if len(df) > 1:
        df.sort_values(by='Num1', inplace=True, ascending=False)
        df_new_sort = df.apply(lambda x:x.iloc[-1])
        return df_new_sort
# 提取首句与空集
def sim_rel_first(df):
    if len(df) == 1:
        # df_new_sort = df.apply(lambda x: x.iloc[0])
        return df
if __name__ == '__main__':
    # 将同一问题下的评论进行笛卡尔积形式组合
    # xlsx_file = r"H:\BERT-CRF-for-NER\Comment_data\Top1-5912_questions_so_comment.xlsx"
    # comment_sheet = pd.read_excel(xlsx_file)
    # comment_row = comment_sheet.loc[0].values
    # print("\n{0}".format(comment_row))
    #
    # comment_sheet = pd.DataFrame(comment_sheet)
    # comment_group = comment_sheet.groupby("question_header", sort=False)
    #
    # ratings = comment_group.apply(sim_rel)
    #
    # ratings_df = pd.DataFrame(ratings)
    # ratings_df.reset_index(inplace=True)
    # ratings_df.to_excel("Top1-4456_senpair.xlsx")

    # 根据预测结果进行排序
    xlsx_file_sort = r"H:\BERT-CRF-for-NER\Comment_data\Top1-4456_senpair_prediction.xlsx"
    comment_sheet_sort = pd.read_excel(xlsx_file_sort)
    # comment_row_sort = comment_sheet_sort.loc[0].values
    # print("\n{0}".format(comment_row_sort))
    # print("\n{0}".format(comment_sheet_sort))
    comment_sheet_sort = pd.DataFrame(comment_sheet_sort)
    # print("\n{0}".format(comment_sheet_sort))
    comment_group_sort = comment_sheet_sort.groupby("string2", sort=False, group_keys=False, as_index=False)
    ratings_sort = comment_group_sort.apply(sim_rel_sort)
    ratings_sort_na = comment_group_sort.apply(sim_rel_nagative_sort)
    ratings_sort_first = comment_group_sort.apply(sim_rel_first)
    ratings_df_sort = pd.DataFrame(ratings_sort)
    ratings_df_sort_na = pd.DataFrame(ratings_sort_na)
    ratings_df_sort_first = pd.DataFrame(ratings_sort_first)

    ratings_df_sort.reset_index(inplace=False)
    ratings_df_sort_na.reset_index(inplace=False)
    ratings_df_sort_first.reset_index(inplace=False)
    ratings_df_sort.to_excel("Top1-4456_senpair_sorted.xlsx")



>>>>>>> 90df61a1eb4949d38c3583627ea6ba339ab9ea7a
