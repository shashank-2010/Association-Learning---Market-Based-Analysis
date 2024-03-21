# Association-Learning---Market-Based-Analysis
Association learning in machine learning uncovers hidden patterns between items in market data. By analyzing purchase history, it helps identify frequently bought together products, allowing businesses to optimize product placement, promotions, and recommendations for stronger sales.

## Libraries and Modules Used
![image](https://github.com/shashank-2010/Association-Learning---Market-Based-Analysis/assets/153171192/9307e7b9-4781-4d18-bf75-72ad9ab81a53)

## Read Data
![image](https://github.com/shashank-2010/Association-Learning---Market-Based-Analysis/assets/153171192/763b0e5d-f2fd-48f7-a403-e480a6da528c)

## Feature Encoding using mlxtend
TransactionEncoder() is designed to transform transactional data, typically represented as a list of lists, into a one-hot encoded format suitable for various machine learning algorithms.
Transactional data often consists of sets of items or events that occur together. For example, in market basket analysis, each transaction might be a list of items purchased by a customer.
### Functionality:
**Input**: It accepts a list of lists, where the outer list represents transactions, and each inner list represents the items (or events) present in that specific transaction.

**Encoding**: It performs one-hot encoding on the data. This means it creates a new sparse matrix or DataFrame where each row represents a transaction and each column represents a unique item (or event) encountered in all transactions.
If an item appears in a transaction, the corresponding value in the encoded matrix/DataFrame will be 1.
If an item doesn't appear in a transaction, the corresponding value will be 0.
This encoding allows machine learning algorithms to understand the relationships between items based on their co-occurrence in transactions.

**Output**: The TransactionEncoder returns the one-hot encoded representation of the transactional data, typically as a sparse matrix (using libraries like SciPy) or a pandas DataFrame.
