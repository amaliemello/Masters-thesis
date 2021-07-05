# This code reads in "twinAndRenal_study.txt".
# It splits the dataset into one text-file with T1D and one with their twin.
# It creates the file "inputML.txt", which can be used as input for machine learning.

import re
count = len(open("twinAndRenal_study.txt").readlines(  ))  # 27580
print(count)
# Opens file and makes the variable lists:
lists = []
with open("twinAndRenal_study.txt", "r") as f:
    tittel = f.readline() # Line 1 is the title
    descriptionsList = f.readline().split("\t") # Line two is the descriptions
    for lis in range(count-2): # Line 3 to 27580
        midlertidigList = f.readline().split("\t")
        matches1 = 0
        pattern1 = re.compile(r"\d.\d+E.\d+")
        match1 = pattern1.finditer(str(midlertidigList))
        lengde = 0
        for m in match1:
            lengde = lengde+1
        if lengde < 1:
            lists.append(midlertidigList)
# Now, lists are all lines from line 3 in the dataset (exept the lines including -3.40E+38),
# starting with the cg number that has index 0.
print(len(lists))

# Making a list with descriptions of those with T1D from twin_study:
T1DDescriptions = []
for object_in_list in descriptionsList:
    text_to_search = object_in_list
    pattern = re.compile(r'"\d+_CD1?4_T1Dpair\d+_T1D:AVG_Beta"\n?')  # defines the pattern that will be searched for.
    matches = pattern.finditer(text_to_search)  # indicates where the pattern will be searched for in.
    for match in matches:
        # print(match[0])  # [0] indicates which property
        T1DDescriptions.append(match[0])

# Making a list with descriptions of those that have a twin with T1D:
T1DTwinDescriptions = []
for object_in_list in descriptionsList:
    text_to_search = object_in_list
    pattern = re.compile(r'"\d+_CD1?4_T1Dpair\d+_unaffected:AVG_Beta"\n?')
    matches = pattern.finditer(text_to_search)
    for match in matches:
        T1DTwinDescriptions.append(match[0])

# Making a list with descriptions of ALL those with T1D:
allT1DDescriptions = []
for object_in_list in descriptionsList:
    text_to_search = object_in_list
    pattern = re.compile(r'"\d+_CD1?4_T1Dpair\d+_T1D:AVG_Beta"\n?')  # defines the pattern that will be searched for.
    matches = pattern.finditer(text_to_search)  # indicates where the pattern will be searched for in.
    for match in matches:
        # print(match[0])  # [0] indicates which property
        allT1DDescriptions.append(match[0])
for object_in_list in descriptionsList:
    text_to_search = object_in_list
    pattern = re.compile(r'"Whole Blood D*C*N*\d*r*"\n?')  # defines the pattern that will be searched for.
    matches = pattern.finditer(text_to_search)  # indicates where the pattern will be searched for in.
    for match in matches:
        # print(match[0])  # [0] indicates which property
        allT1DDescriptions.append(match[0])

# Making a list with descriptions of those WITHOUT T1D:
notT1DDescriptions = []
for object_in_list in descriptionsList:
    text_to_search = object_in_list
    pattern = re.compile(r'"\d+_CD1?4_T1Dpair\d+_unaffected:AVG_Beta"\n?')
    matches = pattern.finditer(text_to_search)
    for match in matches:
        notT1DDescriptions.append(match[0])
for object_in_list in descriptionsList:
    text_to_search = object_in_list
    pattern = re.compile(r'"\d\d\d?_CD1?4_Normalpair\d_normal_MZ:AVG_Beta"\n?')
    matches = pattern.finditer(text_to_search)
    for match in matches:
        notT1DDescriptions.append(match[0])

# Function that allow you to choose indices:
# source: https://www.oreilly.com/library/view/
# python-cookbook/0596001673/ch04s07.html


def select(lst, indices):
    return (lst[i] for i in indices)


print("len(lists[3]:", len(lists[3]))  # 296

# Indices that will be removed because of too many zeros.
antNullList = [0]*(len(lists[3]))

for i in range(len(lists)):
    u = 0
    k = i

    L = lists[i]
    ValuesTuple1 = select(L, list(range(0, len(lists[i]))))  # 101
    ValuesStr1 = ""
    for n in ValuesTuple1:
        u += 1
        if n == "0":
            antNullList[u-1] += 1
print("len(antNullLists):", (antNullList[219]))  # The first one is zero, as it is cg code. (219 is high)
u = 0
indicesToDelete = []
for i in antNullList:
    if i > ((len(lists))*0.05):
        indicesToDelete.append(u)
    u += 1
print(indicesToDelete)

# Find the indices that T1D descriptions have in descriptionsList:
T1DIndexList = [0]  # Starts with zero to get the nucleotide positions first in each row.
for i in range(len(descriptionsList)):
    for j in range(len(T1DDescriptions)):
        if descriptionsList[i] == T1DDescriptions[j]:
            T1DIndexList.append(i)
T1DIndexList = [i for i in T1DIndexList if i not in indicesToDelete]

# Find the indices that T1DTwin descriptions have in descriptionsList:

T1DTwinIndexList = [0]
for i in range(len(descriptionsList)):
    for j in range(len(T1DTwinDescriptions)):
        if descriptionsList[i] == T1DTwinDescriptions[j]:
            T1DTwinIndexList.append(i)
T1DTwinIndexList = [i for i in T1DTwinIndexList if i not in indicesToDelete]

# Find the indices that allT1D descriptions have in descriptionsList:
allT1DIndexList = [0]
for i in range(len(descriptionsList)):
    for j in range(len(allT1DDescriptions)):
        if descriptionsList[i] == allT1DDescriptions[j]:
            allT1DIndexList.append(i)

allT1DIndexList = [i for i in allT1DIndexList if i not in indicesToDelete]

# Find the indices that notT1D descriptions have in descriptionsList:
notT1DIndexList = [0]
for i in range(len(descriptionsList)):
    for j in range(len(notT1DDescriptions)):
        if descriptionsList[i] == notT1DDescriptions[j]:
            notT1DIndexList.append(i)
notT1DIndexList = [i for i in notT1DIndexList if i not in indicesToDelete]

# Find the indices that cg descriptions have in descriptionList:
cgIndexList = [0]

T1DDescriptionsStr = ""
T1DDescriptionsStr += "\n"
T1DTwinDescriptionsStr = ""
T1DTwinDescriptionsStr += "\n"
cgDescriptionsStr = ""
allT1DDescriptionsStr = "\n"
notT1DDescriptionsStr = "\n"

# Write T1D.txt
with open("T1D.txt", "w") as T1D_f:
    # T1D_f.write("Nucleotide position of CpG site - T1D"+"\n")
    T1D_f.write(T1DDescriptionsStr)
    for i in range(len(lists)):
        L = lists[i]
        T1DValuesTuple = select(L, T1DIndexList)
        T1DValuesStr = ""
        for i in T1DValuesTuple:
            T1DValuesStr += i + "\t"
        if T1DValuesStr.endswith("\t"):
            T1DValuesStr = T1DValuesStr[:-1]
        T1DValuesStr += "\n"
        T1DValuesStr = T1DValuesStr.replace('"', '')
        T1DValuesStr = T1DValuesStr.replace('\t\t', '\t0\t')
        T1DValuesStr = T1DValuesStr.replace('\t\t', '\t0\t')
        T1D_f.write(T1DValuesStr)

# write T1DTwin.txt
with open("T1DTwin.txt", "w") as T1DTwin_f:
    # T1DTwin_f.write("Nucleotide position of CpG site - T1D"+"\n")
    T1DTwin_f.write(T1DTwinDescriptionsStr)
    for i in range(len(lists)):
        L = lists[i]
        T1DTwinValuesTuple = select(L, T1DTwinIndexList)
        T1DTwinValuesStr = ""
        for i in T1DTwinValuesTuple:
            T1DTwinValuesStr += i + "\t"
        if T1DTwinValuesStr.endswith("\t"):
            T1DTwinValuesStr = T1DTwinValuesStr[:-1]
        T1DTwinValuesStr += "\n"

        T1DTwinValuesStr = T1DTwinValuesStr.replace('"', '')
        T1DTwinValuesStr = T1DTwinValuesStr.replace('\t\t', '\t0\t')
        T1DTwinValuesStr = T1DTwinValuesStr.replace('\t\t', '\t0\t')
        T1DTwin_f.write(T1DTwinValuesStr)

# write cg.txt
with open("cg.txt", "w") as cg_f:
    # cg_f.write("Nucleotide position of CpG site - T1D"+"\n")
    cg_f.write(cgDescriptionsStr)
    for i in range(len(lists)):
        L = lists[i]
        cgValuesTuple = select(L, cgIndexList)
        cgValuesStr = ""
        for i in cgValuesTuple:
            cgValuesStr += i + "\t"
        if cgValuesStr.endswith("\t"):
            cgValuesStr=cgValuesStr[:-1]
        cgValuesStr += "\n"
        cgValuesStr = cgValuesStr.replace('"', '')
        cgValuesStr = cgValuesStr.replace('\t\t', '\t0\t')
        cgValuesStr = cgValuesStr.replace('\t\t', '\t0\t')
        cg_f.write(cgValuesStr)

# write allT1D.txt
with open("allT1D.txt", "w") as allT1D_f:
    allT1D_f.write(allT1DDescriptionsStr)
    for i in range(len(lists)):
        L = lists[i]
        allT1DValuesTuple = select(L, allT1DIndexList)
        allT1DValuesStr = ""
        for i in allT1DValuesTuple:
            allT1DValuesStr += i + "\t"
        if allT1DValuesStr.endswith("\t"):
            allT1DValuesStr = allT1DValuesStr[:-1]
        allT1DValuesStr += "\n"
        allT1DValuesStr = allT1DValuesStr.replace('"', '')
        allT1DValuesStr = allT1DValuesStr.replace('\t\t', '\t0\t')
        allT1DValuesStr = allT1DValuesStr.replace('\t\t', '\t0\t')
        allT1D_f.write(allT1DValuesStr)

# write notT1D.txt
with open("notT1D.txt", "w") as notT1D_f:
    notT1D_f.write(notT1DDescriptionsStr)
    for i in range(len(lists)):
        L = lists[i]
        notT1DValuesTuple = select(L, notT1DIndexList)
        notT1DValuesStr = ""
        for i in notT1DValuesTuple:
            notT1DValuesStr += i + "\t"
        if notT1DValuesStr.endswith("\t"):
            notT1DValuesStr = notT1DValuesStr[:-1]
        notT1DValuesStr += "\n"
        notT1DValuesStr = notT1DValuesStr.replace('"', '')
        notT1DValuesStr = notT1DValuesStr.replace('\t\t', '\t0\t')
        notT1DValuesStr = notT1DValuesStr.replace('\t\t', '\t0\t')
        notT1D_f.write(notT1DValuesStr)

# write inputML.txt
allIndexList = allT1DIndexList[1:]+notT1DIndexList[1:]  # 226 T1D + 68 NOT T1D (295 in total)

with open("inputML.txt", "w") as inputML_f:
    inputML_f.write("\n226 T1D and then 68 not T1D \n")
    antNullList = [0]*len(allIndexList)
    for i in range(len(lists)):
        u = 0
        L = lists[i]
        allValuesTuple = select(L, allIndexList)
        allValuesStr = ""
        t = 0
        for j in allValuesTuple:
            t += 1
            if t == 1:
                if j == "\t":
                    print("warning: write 'inputML.txt'")
            allValuesStr += j + "\t"
            u += 1
            if j == "0":
                antNullList[u-1] += 1
        if allValuesStr.endswith("\t"):
            allValuesStr = allValuesStr[:-1]
        allValuesStr = allValuesStr.replace("\n", "")
        allValuesStr += "\n"
        allStr = allValuesStr.replace('"', '')  # Not in use?
        inputML_f.write(allValuesStr)

    p = 0
    for i in antNullList:
        p += 1
        if i > (len(lists))/2:
            print(p)
    print(allValuesStr)
    print(len(antNullList))
    print(antNullList[0])
    #print("person 1 zeros:", person1Zeros)

countML = len(open("inputML.txt").readlines(  ))
# If around 54013, each line are split in too, with has to be corrected.
print("ML:", countML, "lines. The", len(allT1DIndexList[1:]), "first are T1D, and the", len(notT1DIndexList[1:]), "next are not T1D.")

nullIndexList = [0,  219]

#write null.txt
with open("null.txt", "w") as null_f:
    null_f.write(T1DDescriptionsStr)
    for i in range(len(lists)):
        L = lists[i]
        nullValuesTuple = select(L, nullIndexList)
        nullValuesStr = ""
        for i in nullValuesTuple:
            nullValuesStr +=i + "\t"
        if nullValuesStr.endswith("\t"):
            nullValuesStr=nullValuesStr[:-1]
        nullValuesStr += "\n"
        nullValuesStr = nullValuesStr.replace('"', '')
        nullValuesStr = nullValuesStr.replace('\t\t', '\t0\t')
        nullValuesStr = nullValuesStr.replace('\t\t', '\t0\t')
        null_f.write(nullValuesStr)
