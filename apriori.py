import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#inline dataset
dataset =[['Noodles', 'Pickles', 'Milk'],
          ['Noodles','Cheese'],
          ['Cheese', 'Shoes'],
          ['Noodles', 'Pickles','Cheese'],
          ['Noodles', 'Pickles','Clothes','Cheese','Milk'],
          ['Pickles','Clothes','Milk'],
          ['Pickles','Milk','Clothes']]

#TransactionEncoder to convert from categorical to numeric (6 items so 0-5) 
#and using True if item present in transaction, False otherwise
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)

#loading the transaction data into Dataframe
df = pd.DataFrame(te_array,columns=te.columns_)
print(df)

print("\n\n")

#applying apriori algorithm to get frequent itemsets
freq_itemsets= apriori(df,min_support=0.3, use_colnames=True)
print(freq_itemsets)

print("\n\n")

#applying association_rules to get the association rules based on confidence
assoc_rules=association_rules(freq_itemsets,metric= "confidence",min_threshold=0.80)

print(assoc_rules)

