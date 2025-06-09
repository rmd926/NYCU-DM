import sys
import time
from collections import defaultdict
from optparse import OptionParser

def dataFromFile(fname):
    """Read transaction data from file and return a generator of ordered items."""
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")
            tokens = line.split()[3:]  # Assumes the transaction starts from the 4th element
            yield tokens  # Return list to keep order

class FPTree:
    def __init__(self):
        """Initialize FP-Tree"""
        self.root = {'name': None, 'count': None, 'parent': None, 'children': {}}
        self.headers = defaultdict(list)

    def add_transaction(self, transaction, count=1):
        """Add a transaction to the FP-Tree preserving the original order."""
        current_node = self.root
        for item in transaction:  # transaction is a list preserving the order
            if item in current_node['children']:
                current_node['children'][item]['count'] += count
            else:
                new_node = {'name': item, 'count': count, 'parent': current_node, 'children': {}}
                current_node['children'][item] = new_node
                self.headers[item].append(new_node)
            current_node = current_node['children'][item]

    def mine_patterns(self, min_support):
        """Mine frequent patterns"""
        patterns = {}
        for item, nodes in self.headers.items():
            support = sum(node['count'] for node in nodes)
            if support >= min_support:
                patterns[frozenset([item])] = support
                self._mine_conditional_tree(item, min_support, patterns, suffix=[item])
        return patterns

    def _mine_conditional_tree(self, item, min_support, patterns, suffix):
        """Recursively mine conditional pattern tree"""
        conditional_tree = FPTree()
        for node in self.headers[item]:
            path = []
            parent = node['parent']
            while parent and parent['name']:
                path.append(parent['name'])
                parent = parent['parent']
            conditional_tree.add_transaction(path, node['count'])

        pruned_patterns = conditional_tree.mine_patterns(min_support)
        for new_item, new_support in pruned_patterns.items():
            if new_support >= min_support:
                new_pattern = suffix + list(new_item)
                patterns[frozenset(new_pattern)] = new_support

        return patterns

def runFPGrowth(transactions, minSupport):
    """Run FP-Growth algorithm and generate frequent itemsets"""
    tree = FPTree()
    for transaction in transactions:
        tree.add_transaction(sorted(transaction))
    return tree.mine_patterns(minSupport * len(transactions))

def write_itemsets_to_file(itemsets, filename, total_transactions):
    """Write frequent itemsets to file preserving the order within itemsets."""
    with open(filename, 'w') as f:
        for itemset, support in sorted(itemsets.items(), key=lambda x: -x[1]):  # Sort by support
            support_percent = (support / total_transactions) * 100
            f.write(f"{support_percent:.1f}\t{{{', '.join(itemset)}}}\n")

# Main program
if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option("-f", "--inputFile", dest="input", help="Input file containing dataset", default='datasetA.data')
    optparser.add_option("-s", "--minSupport", dest="minS", help="Minimum support value", default=0.1, type="float")
    optparser.add_option("-p", "--step", dest="step", help="Step number (2 or 3)", default='3')

    (options, args) = optparser.parse_args()

    if options.input is None:
        sys.exit("Dataset file must be provided.")
    else:
        transactions = list(dataFromFile(options.input))
        total_transactions = len(transactions)

    dataset_name = options.input.split('.')[0]
    minSupport = options.minS

    # Task 1: Frequent itemset mining
    start_time_task1 = time.perf_counter()
    frequent_itemsets = runFPGrowth(transactions, minSupport)
    task1_itemsets_file = f"step{options.step}_task1_{dataset_name}_{minSupport:.3f}_result1.txt"
    write_itemsets_to_file(frequent_itemsets, task1_itemsets_file, total_transactions)
    
    end_time_task1 = time.perf_counter()
    task1_time = end_time_task1 - start_time_task1

    print(f"Task 1 computation time: {task1_time:.6f} seconds")

    print("files generated.")

