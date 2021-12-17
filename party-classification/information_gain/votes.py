import dtree

tree = dtree.dtree()
train, test, classes, test_classes, features, dataset_length = tree.read_data('processed_data.data')
t = tree.make_tree(train, classes, features)
tree.printTree(t, ' ')

print("Train data classes")
print(classes)

print("Train data classified classes")
print(tree.classifyAll(t, train))

for i in range(len(train)):
    tree.classify(t, train[i])

print("Test data classes")
print(test_classes)

print("Test data")
test_classification = tree.classifyAll(t, test)
print(test_classification)

for i in range(len(test)):
    tree.classify(t, test[i])

match_count = 0
for i in range(len(test_classes)):
    if test_classes[i] != test_classification[i]:
        match_count += 1

test_error = match_count/len(test_classes)
print("Test error: {}".format(test_error))
