# This code creates twinAndRenal_study.txt
# - a file with an empty line, and then genenames on the first column and values from first twin_study.txt
# and then from renal_study.txt.

import re

# Function that allow you to choose indices:
# source: https://www.oreilly.com/library/view/
# python-cookbook/0596001673/ch04s07.html


def select(lst, indices):
    return (lst[i] for i in indices)


count = len(open("twin_study.txt").readlines(  ))  # 27580
print("Number of lines in twin_study.txt:", count)
# Opens file and makes the variable lists:
lists1 = []
with open("twin_study.txt", "r") as twin_f:
    tittel1 = twin_f.readline()  # Line 1 is the title
    descriptionsList1 = twin_f.readline().split("\t")  # Line two is the descriptions
    for lis in range(count-2):  # Line 3 to 27580
        midlertidigList1 = twin_f.readline().split("\t")
        lists1.append(midlertidigList1)

# Now, lists1 are all lines from line 3 in the dataset , starting with the cg number that has index 0.

count = len(open("renal_study.txt").readlines(  ))  # 27580
print("Number of lines in renal_study.txt:", count)
# Opens file and makes the variable lists:
lists2 = []
with open("renal_study.txt", "r") as renal_f:
    tittel2 = renal_f.readline()  # Line 1 is the title
    descriptionsList2 = renal_f.readline().split("\t")  # Line two is the descriptions
    for lis in range(count-2):  # Line 3 to 27580
        midlertidigList2 = renal_f.readline().split("\t")
        lists2.append(midlertidigList2)
print(descriptionsList2)

descriptionList = []
for i in descriptionsList1:
    i = i.replace("\n", "")
    descriptionList.append(i)
for j in descriptionsList2[1:]:
    descriptionList.append(j)

# Now, zeros will be counted and final file will be made.
with open("twinAndRenal_study.txt", "w") as twinAndRenal_study_f:
    twinAndRenal_study_f.writelines(["twinAndRenal_study\n"])
    # writing descriptions:
    for item in descriptionList[:-1]:
        twinAndRenal_study_f.write("%s\t" %item)
    for item in descriptionList[-1]:
        twinAndRenal_study_f.write("%s" %item)
    antNullList1 = [0]*(len((lists1[3]))-1)

    for i in range(len(lists1)):
        k = i
        L = lists1[i]
        ValuesTuple1 = select(L, list(range(0,len(lists1[i]))))  # 101
        ValuesStr1 = ""
        for n in ValuesTuple1:
            ValuesStr1 += n + "\t"
        if ValuesStr1.endswith("\n"):
            ValuesStr1 = ValuesStr1[:-2]
        if ValuesStr1.endswith("\t"):
            ValuesStr1 = ValuesStr1[:-2]

        M = lists2[k]
        ValuesTuple2 = select(M, list(range(1, len(lists2[k]))))  # 196
        ValuesStr2 = "\t"
        for j in ValuesTuple2:
            ValuesStr2 += j + "\t"
        ValuesStr = ValuesStr1 + ValuesStr2
        if ValuesStr.endswith("\t"):
            ValuesStr=ValuesStr[:-1]
        #ValuesStr += "\n"
        #ValuesStr = ValuesStr.replace('"', '')
        ValuesStr = ValuesStr.replace('\t\t', '\t0\t')
        ValuesStr = ValuesStr.replace('\t\t', '\t0\t')
        ValuesStr = ValuesStr.replace('null', '0')
        twinAndRenal_study_f.write(ValuesStr)
count = len(open("twinAndRenal_study.txt").readlines(  ))
print("Number of lines in twinAndRenal_study.txt:", count)
