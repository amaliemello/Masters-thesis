#This code creates a dictionary with CpG sites as keys and gene names as values.
# The code also reads in files with CpG sites and writes files with associated gene names.

count = len(open("cgToGenename.txt").readlines(  ))
print(count)
#Opens file and makes the variable lists:
lists = []
with open("cgToGeneName.txt", "r") as f:

    #descriptionsList= f.readline().split("\t") #Line two is the descriptions
    for lis in range(count): #Line 1 to the last,
        midlertidigList = f.readline().split("\t")
        lists.append(midlertidigList)

# Now, lists are all lines from line 3 in the dataset, starting with the cg number that has index 0.
print(len(lists))

#Function that allow you to choose indeces:
#source: https://www.oreilly.com/library/view/
# python-cookbook/0596001673/ch04s07.html
def select(lst, indices):
    return (lst[i] for i in indices)

# Make dictionary:
cgDict= {}
for i in range(len(lists)):
    L=lists[i]
    valuesTuple1 = select(L, [0])
    valuesTuple2 = select(L, [21])
    for i in valuesTuple1:
        for j in valuesTuple2:
            cgDict[i] = j
# print(cgDict['cg10031456'])

# open cg.txt:
count2 = len(open("cg.txt").readlines())#
print(count2)
# Opens file and makes the variable lists:
lists2 = []
with open("cg.txt", "r") as cg_f:
    for lis in range(count2):  # Line 1 to the last, but not the very last that is empty
        midlertidigList2 = cg_f.readline()
        midlertidigList2 = midlertidigList2[:-1]
        #print(midlertidigList2)
        lists2.append(midlertidigList2)

# open Neighbours_cg09736162.txt:
count62 = len(open("Neighbours_cg09736162.txt").readlines())#
# Opens file and makes the variable lists:
lists62 = []
with open("Neighbours_cg09736162.txt", "r") as Neighbours_cg09736162_f:
    for lis in range(count62):  # Line 1 to the last, but not the very last that is empty
        midlertidigList62 = Neighbours_cg09736162_f.readline()
        midlertidigList62 = midlertidigList62[:-1]
        lists62.append(midlertidigList62)

# open Neighbours_cg23173455.txt:
count55 = len(open("Neighbours_cg23173455.txt").readlines())#
# Opens file and makes the variable lists:
lists55 = []
with open("Neighbours_cg23173455.txt", "r") as Neighbours_cg23173455_f:
    for lis in range(count55):  # Line 1 to the last, but not the very last that is empty
        midlertidigList55 = Neighbours_cg23173455_f.readline()
        midlertidigList55 = midlertidigList55[:-1]
        lists55.append(midlertidigList55)

#open Neighbours_cg04542415.txt:
count15 = len(open("Neighbours_cg04542415.txt").readlines())#
#Opens file and makes the variable lists:
lists15 = []
with open("Neighbours_cg04542415.txt", "r") as Neighbours_cg04542415_f:
    for lis in range(count15):  #Line 1 to the last, but not the very last that is empty
        midlertidigList15 = Neighbours_cg04542415_f.readline()
        midlertidigList15 = midlertidigList15[:-1]
        lists15.append(midlertidigList15)

#cg10031456
#open Neighbours_cg10031456.txt:
count56 = len(open("Neighbours_cg10031456.txt").readlines())#
#Opens file and makes the variable lists:
lists56 = []
with open("Neighbours_cg10031456.txt", "r") as Neighbours_cg10031456_f:
    for lis in range(count56): #Line 1 to the last, but not the very last that is empty
        midlertidigList56 = Neighbours_cg10031456_f.readline()
        midlertidigList56 = midlertidigList56[:-1]
        lists56.append(midlertidigList56)

#Now, lists2 are all lines from line 1 in the dataset, sincluding only the cg number
print("Length og list2: ", len(lists2))
print("list2[4]= ", lists2[4])
example = lists2[0]
print("example:" ,cgDict[example])

#write geneNamesNeighbours_cg09736162.txt
with open("geneNamesNeighbours_cg09736162.txt", "w") as geneNamesNeighbours_cg09736162_f:
    geneNamesNeighbours_cg09736162 = ""
    cgNotInDict62 = 0
    for i in range(len(lists62)):
        holder62 = lists62[i]
        if holder62 in cgDict.keys():
            geneNameNeighbours_cg09736162 = cgDict[holder62]
            geneNameNeighbours_cg09736162 += "\n"
            geneNamesNeighbours_cg09736162 += geneNameNeighbours_cg09736162
        else:
            cgNotInDict62 += 1
    geneNamesNeighbours_cg09736162_f.write\
        (geneNamesNeighbours_cg09736162)
    print(cgNotInDict62)

#write geneNamesNeighbours_cg23173455.txt
with open("geneNamesNeighbours_cg23173455.txt", "w") as geneNamesNeighbours_cg23173455_f:
    geneNamesNeighbours_cg23173455 = ""
    cgNotInDict55 = 0
    for i in range(len(lists55)):
        holder55 = lists55[i]
        if holder55 in cgDict.keys():
            geneNameNeighbours_cg23173455 = cgDict[holder55]
            geneNameNeighbours_cg23173455 += "\n"
            geneNamesNeighbours_cg23173455 += geneNameNeighbours_cg23173455
        else:
            cgNotInDict55 += 1
    geneNamesNeighbours_cg23173455_f.write\
        (geneNamesNeighbours_cg23173455)
    print(cgNotInDict55)

#write geneNamesNeighbours_cg04542415.txt
with open("geneNamesNeighbours_cg04542415.txt", "w") as geneNamesNeighbours_cg04542415_f:
    geneNamesNeighbours_cg04542415 = ""
    cgNotInDict15 = 0
    for i in range(len(lists15)):
        holder15 = lists15[i]
        if holder15 in cgDict.keys():
            geneNameNeighbours_cg04542415 = cgDict[holder15]
            geneNameNeighbours_cg04542415 += "\n"
            geneNamesNeighbours_cg04542415 += geneNameNeighbours_cg04542415
        else:
            cgNotInDict15 += 1
    geneNamesNeighbours_cg04542415_f.write\
        (geneNamesNeighbours_cg04542415)
    print(cgNotInDict15)

#10031456
#write geneNamesNeighbours_cg10031456.txt
with open("geneNamesNeighbours_cg10031456.txt", "w") as geneNamesNeighbours_cg10031456_f:
    geneNamesNeighbours_cg10031456 = ""
    cgNotInDict56 = 0
    for i in range(len(lists56)):
        holder56 = lists56[i]
        if holder56 in cgDict.keys():
            geneNameNeighbours_cg10031456 = cgDict[holder56]
            geneNameNeighbours_cg10031456 += "\n"
            geneNamesNeighbours_cg10031456 += geneNameNeighbours_cg10031456
        else:
            cgNotInDict56 += 1
    geneNamesNeighbours_cg10031456_f.write\
        (geneNamesNeighbours_cg10031456)
    print(cgNotInDict56)

#write geneNames.txt (ALL)
with open("geneNames.txt", "w") as geneNames_f:
    geneNames = ""
    cgNotInDict = 0
    for i in range(len(lists2)):
        holder = lists2[i] #ok:cg27665659

        if holder in cgDict.keys():
            geneName = cgDict[holder]
            geneName += "\n"
            geneNames += geneName
        else:
            cgNotInDict += 1
            geneName = "no name"
            geneName += "\n"
            geneNames += geneName

    geneNames_f.write(geneNames)
    print('CpG sites that are not found in dict.: ', cgNotInDict)