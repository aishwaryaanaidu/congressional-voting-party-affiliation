## Code to run the decision tree on the Party dataset ##

# Implemented by Stephen Marsland
# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.


import dtree

tree = dtree.dtree()
party,classes,features = tree.read_data('processed_data.data')
t=tree.make_tree(party,classes,features)
tree.printTree(t,' ')

print("Classified classes")
classification = tree.classifyAll(t,party)
print(classification)

for i in range(len(party)):
    tree.classify(t,party[i])


print("True Classes")
print(classes)

match_count = 0
for i in range(len(classes)):
    if classes[i] != classification[i]:
        match_count += 1

error = match_count/len(classes)
print("Error: {}".format(error))
