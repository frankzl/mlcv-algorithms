# Gini Index
# evaluates splits in the dataset
# perfect separation => Gini index 0
# 50/50 in each group => Gini index 0.5

# proportion of classes in each group
proportion = count(class_value) / count(rows)

# Gini calculated for each child node
gini_index = sum( proportion * (1.0 - proportion) )
gini_index = 1.0 - sum( proportion * proportion )
