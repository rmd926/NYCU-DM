"""
Description     : Simple Python implementation of the Apriori Algorithm
Modified from:  https://github.com/asaini/Apriori
Usage:
    $python apriori.py -f DATASET.csv -s minSupport

    $python apriori.py -f DATASET.csv -s 0.15
"""

import sys
import time
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)
    
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet

# 得到itemset的組合
def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList

def is_closed_itemset(item, itemsets, freqSet):
    for larger_item in itemsets:
        if item < larger_item and freqSet[item] == freqSet[larger_item]:
            return False
    return True

def findClosedItemsets(itemsets, freqSet):
    closed_itemsets = []
    for item in itemsets:
        if is_closed_itemset(item, itemsets, freqSet):
            closed_itemsets.append(item)
    print(f"Number of closed itemsets found: {len(closed_itemsets)}")
    return closed_itemsets

def runApriori(data_iter, minSupport):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport
    stats = {'iterations': {}, 'total_frequent_itemsets': 0}
    
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    
    currentLSet = oneCSet
    k = 2
    iteration_count = 1
    
    while currentLSet != set([]):    
        largeSet[k - 1] = currentLSet
        before_pruning = len(currentLSet)
        
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet, transactionList, minSupport, freqSet)
        
        after_pruning = len(currentCSet)
        stats['iterations'][iteration_count] = {'before_pruning': before_pruning, 'after_pruning': after_pruning}
        
        currentLSet = currentCSet
        k = k + 1
        iteration_count += 1

    stats['total_frequent_itemsets'] = sum(len(v) for v in largeSet.values())

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    return toRetItems, stats, freqSet
'''
###revise!!!###

def printResults(items):
    """prints the generated itemsets sorted by support """
    for item, support in sorted(items, key=lambda x: x[1]):
        print("item: %s , %.3f" % (str(item), support))


def to_str_results(items):
    """prints the generated itemsets sorted by support"""
    i = []
    for item, support in sorted(items, key=lambda x: x[1]):
        x = "item: %s , %.3f" % (str(item), support)
        i.append(x)
    return i

'''
def write_itemsets_to_file(itemsets, filename):
    with open(filename, 'w') as f:
        for item, support in sorted(itemsets, key=lambda x: -x[1]):
            f.write(f"{support*100:.1f}\t{{{', '.join(map(str, item))}}}\n")

def write_statistics_to_file(stats, filename):
    with open(filename, 'w') as f:
        f.write(f"{stats['total_frequent_itemsets']}\n")
        for iteration, data in stats['iterations'].items():
            f.write(f"{iteration}    {data['before_pruning']}\t{data['after_pruning']}\n")

def write_closed_itemsets_to_file(closed_itemsets, filename, freqSet, transactionList_length):
    with open(filename, 'w') as f:
        f.write(f"{len(closed_itemsets)}\n")
        for item in sorted(closed_itemsets, key=lambda x: -freqSet[frozenset(x)]):
            support = (freqSet[frozenset(item)] / transactionList_length) * 100
            f.write(f"{support:.1f}\t{{{', '.join(map(str, item))}}}\n")

def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")  # Remove trailing comma
            tokens = line.split()
            record = frozenset(tokens[3:])  #cuz the first three nums will not be included.
            yield record

if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option("-f", "--inputFile", dest="input", help="Filename containing the dataset", default='datasetA.data')
    optparser.add_option("-s", "--minSupport", dest="minS", help="Minimum support value", default=0.1, type="float")
    optparser.add_option("-p", "--step", dest="step", help="Step number (2 or 3)", default='2')

    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
        print("No dataset file specified, system will exit\n")
        sys.exit("System will exit")
    else:
        inFile = list(dataFromFile(options.input))

    dataset_name = options.input.split('.')[0]
    min_support = f"{options.minS:.3f}"
    step = options.step

    task1_itemsets_file = f"step{step}_task1_{dataset_name}_{min_support}_result1.txt"
    task1_stats_file = f"step{step}_task1_{dataset_name}_{min_support}_result2.txt"
    task2_closed_itemsets_file = f"step{step}_task2_{dataset_name}_{min_support}_result1.txt"

    minSupport = options.minS

    # Task 1: Frequent itemset mining
    start_time_task1 = time.perf_counter()
    itemsets, stats, freqSet = runApriori(inFile, minSupport)
    write_itemsets_to_file(itemsets, task1_itemsets_file)
    write_statistics_to_file(stats, task1_stats_file)
    end_time_task1 = time.perf_counter()
    task1_time = end_time_task1 - start_time_task1

    # Task 2: Closed itemset filtering
    start_time_task2 = time.perf_counter()
    closed_itemsets = findClosedItemsets([frozenset(i[0]) for i in itemsets], freqSet)
    write_closed_itemsets_to_file(closed_itemsets, task2_closed_itemsets_file, freqSet, len(inFile))
    end_time_task2 = time.perf_counter()
    task2_time = end_time_task2 - start_time_task2

    total_time = task1_time + task2_time
    ratio = (task2_time / task1_time) * 100
    ovr_ratio = (total_time / task1_time) * 100
    # Display results
    print(f"Task 1 computation time in Apriori: {task1_time:.6f} seconds")
    print(f"Task 2 (Only Searching closed Itemset) computation time in Apriori: {task2_time:.6f} seconds")
    print(f"Total computation time in Apriori: {total_time:.6f} seconds\n")
    print("---------------------------------------Time Ratio will be shown below---------------------------------------\n")
    print(f"[Task 2 (Only Searching closed Itemsets) / Task 1) computation time ratio]: {ratio:.6f}%")
    print(f"[Task 2 (Includes the operations of Task1) / Task 1) computation time ratio]: {ovr_ratio:.6f}%")
