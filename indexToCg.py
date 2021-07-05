# The code reads in indices separated by comma and print list with associated cg names.

indices = 830, 1705, 2035, 3114, 21820, 22988, 23822, 23859, 25568

indices = list(indices)

count = len(open("cg.txt").readlines(  ))
print(count)
#Opens file and makes the variable lists:
lists = []
with open("cg.txt", "r") as f:

    #descriptionsList= f.readline().split("\t") #Line two is the descriptions
    for lis in range(count):  #Line 1 to the last
        midlertidigList = f.readline().split("\t")
        lists.append(midlertidigList)

cgList = []
cgList2 = []
for i in range(len(lists)):
    cgList.append(lists[i])

r = (cgList[27005])
r = str(r)
r = r.replace("['","")
r = r.replace(r"\n']", "")

for i in indices:
    r = cgList[i-1]
    r = str(r)
    r = r.replace("['", "")
    r = r.replace(r"\n']", "")
    cgList2.append(r)
cgList2 = ( '[%s]' % ', '.join(map(str, cgList2)))
print(cgList2)
