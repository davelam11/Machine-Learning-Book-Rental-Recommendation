# Machine-Learning-Book-Rental-Recommendation
Book Rent is the largest online and offline book rental chain in India. Lately, the company has been losing its user base. The main reason for this is that users are not able to choose the right books for themselves. The company wants to solve this problem and increase its revenue and profit. 

[Dataset description]

[BX-Users] It contains the information of users.

user_id - These have been anonymized and mapped to integers

Location - Demographic data is provided

Age - Demographic data is provided

If available, otherwise, these fields contain NULL-values.

 

[BX-Books] 

isbn - Books are identified by their respective ISBNs. Invalid ISBNs have already been removed from the dataset.

book_title

book_author

year_of_publication

publisher


 

[BX-Book-Ratings] Contains the book rating information. 

user_id

isbn

rating - Ratings (`Book-Rating`) are either explicit, expressed on a scale from 1–10 (higher values denoting higher appreciation), or implicit, expressed by 0.
 
========================================================================

Following operations should be performed:

Read the books dataset and explore it

Clean up NaN values

Read the data where ratings are given by users

Take a quick look at the number of unique users and books

Convert ISBN variables to numeric numbers in the correct order

Convert the user_id variable to numeric numbers in the correct order

Convert both user_id and ISBN to the ordered list, i.e., from 0...n-1

Re-index the columns to build a matrix

Split your data into two sets (training and testing)

Make predictions based on user and item variables

Use RMSE to evaluate the predictions
