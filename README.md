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

![image](https://github.com/shashank-2010/Association-Learning---Market-Based-Analysis/assets/153171192/2ef29886-2b8f-414d-a172-dff0c82de234)

## COnverting the matrix into DataFrame for training the model
![image](https://github.com/shashank-2010/Association-Learning---Market-Based-Analysis/assets/153171192/2126bb05-80c1-4378-a2b5-c39d30b71b5b)

## Model Trainig
![image](https://github.com/shashank-2010/Association-Learning---Market-Based-Analysis/assets/153171192/76b270f3-b73d-4be9-a83f-095ad0d2a9a4)

In machine learning, Apriori is a classical algorithm used for association rule learning. It's particularly useful for market basket analysis, where you want to discover frequent itemsets and the relationships between them in transactional data.

Here's a breakdown of Apriori:

### Functionality:

**Input**: Apriori takes a dataset of transactions as input. Each transaction is a list of items (or events) that occurred together. For example, in a grocery store scenario, a transaction might be a list of items purchased by a customer in a single shopping trip.

**Frequent Itemset Mining**: The core of Apriori lies in finding frequent itemsets. These are groups of items that appear together frequently in a significant portion of the transactions. Apriori uses an iterative approach, starting with finding frequent sets of single items (e.g., milk, bread) and then progressively building larger sets (e.g., milk, bread, butter) based on the frequent sets from the previous step.

**Association Rules**: Once frequent itemsets are identified, Apriori generates association rules that express the relationships between these items. These rules typically follow the format "if A, then B," where A and B are frequent itemsets. The strength of an association rule is measured by its support and confidence.

**Support**: This metric indicates how frequently the entire rule (A and B) appears together in the data. A high support value suggests a relevant rule that applies to a significant portion of the transactions.

**Confidence**: This metric represents the conditional probability of finding B in a transaction given that A is already present. A high confidence value suggests that the presence of A strongly suggests the presence of B.
One drawback of the confidence measure is that it might misrepresent the importance of an association. 

**Lift**: This says how likely item B is purchased when item A is purchased, while controlling for how popular item A is.

## Understanding the output after applying association rules
![image](https://github.com/shashank-2010/Association-Learning---Market-Based-Analysis/assets/153171192/166019cd-4c42-4cba-8c1f-c3d72deb1c37)

![image](https://github.com/shashank-2010/Association-Learning---Market-Based-Analysis/assets/153171192/4333bcee-593c-405b-8709-84f42360dca3)

1. Antecedent (Left-Hand Side, LHS): This represents a set of items or conditions that appear on the left side of the association rule. It represents the "if" part of the rule.

2. Consequent (Right-Hand Side, RHS): This represents a single item or event that appears on the right side of the association rule. It represents the "then" part of the rule.  For example, the rule "bread, butter => milk" means "if a customer buys bread and butter, then they are also likely to buy milk."

For example :- The first row can be read as, if soup is bought then custmer is highly likely(*confidence = 0.46, meaning aroung 46% chances*) to by mineral water. 
