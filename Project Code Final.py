import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pickle import dump

#Importing machine learning related functions
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

#Regression functions
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

pd.options.display.width=None
pd.options.display.max_columns=None
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 20)

"""
Step 1: Datashape of CSV Files
"""
print("========== Step 1: Datashape of CSV Files ==========")
books_df = pd.read_csv("BX-Books.csv", encoding="latin1", low_memory=False)
users_df = pd.read_csv("BX-Users.csv", encoding='latin1', low_memory=False)
ratings_df = pd.read_csv("BX-Book-Ratings.csv", encoding='latin1', low_memory=False)

print("Shape of Books Dataset: ", books_df.shape, "// Which means", books_df.shape[0], "rows and", books_df.shape[1], "columns")
print("Shape of Users Dataset: ", users_df.shape, "// Which means", users_df.shape[0], "rows and", users_df.shape[1], "columns")
print('Shape of Ratings Dataset: ', ratings_df.shape, "// Which means", ratings_df.shape[0], "rows and", ratings_df.shape[1], "columns")

"""
Step 2: Clean up NaN Values
"""
print("========== Step 2: Clean up NaN Values ==========")
#-----Missing Value Count-----#
df_list = [books_df,users_df,ratings_df]
df_name_list = ["books_df","users_df","ratings_df"]

for df, df_name in zip(df_list, df_name_list):
    var_null_list = []
    print("For the <{}>, the null value counts are as follow:".format(df_name))
    for column in df.columns:
        a = df[column].isnull().values.any()
        b = df[column].isnull().sum()
        if b > 0:
            var_null_list.append(str(column))
            print(column, ": This variable have null values of " , b, "in total.")
        else:
            print(column, ": This variable don't have any null value.")
    leng = len(var_null_list)
    print("There is in total ", leng, " variables with null values in <{}>".format(df_name))
    if leng > 0:
        print(var_null_list)
    print("----------------------------------------------------------------")

#-----Dropna of Rows with Missing Values-----#
#-----From the above code, we know that:
#-----For books_df, ['book_author', 'publisher'] have null values
#-----For users_df, ['Location', 'Age'] have null values
#-----For ratings_df, No columns have null values
books_df = books_df.dropna(subset=["book_author","publisher"])
users_df = users_df.dropna(subset=["Location","Age"])

"""
Step 3: Unique Count of Users and Books
"""
print("========== Step 3: Unique Count of Users and Books ==========")
unique_book_list = books_df["isbn"].unique()
unique_user_list = users_df["user_id"].unique()
unique_ratings_user_list = ratings_df["user_id"].unique()
print("Number of Unique Books: ", len(unique_book_list))
print("Number of Unique Users: ", len(unique_user_list))
print("Number of Unique Users who had given ratings: ", len(unique_ratings_user_list))

"""
Step 4: Convert ISBN to numeric numbers in the correct order
"""
print("=========== Step 4: Convert ISBN to numeric numbers in the correct order ==========")
#As the last digit of ISBN is a check digit, we need to cut off the last digit to do the correct sorting.
#Processing the isbn column in Books dataset
strip_isbn_list = []
for isbn in books_df.loc[:,"isbn"]:
    strip_isbn = isbn.rstrip(isbn[-1])
    strip_isbn_list.append(strip_isbn)

books_df.insert(1,"stripped_isbn",strip_isbn_list)
books_df = books_df[books_df["stripped_isbn"].astype(str).str.isnumeric() == True]

num_isbn_list = []
for isbn in books_df.loc[:,"stripped_isbn"]:
    num_isbn_list.append(int(isbn))

books_df.insert(2,"numeric_isbn",num_isbn_list)
books_df = books_df.drop(columns=["isbn","stripped_isbn"])

#Extra step: Convert year of publication column into Integer
books_df = books_df[books_df["year_of_publication"].astype(str).str.isnumeric() == True]

num_year_list = []
for year in books_df.loc[:,"year_of_publication"]:
    num_year_list.append(int(year))

books_df.insert(5, "pub_year", num_year_list)
books_df = books_df.drop(columns=["year_of_publication"])
sorted_books_df = books_df.sort_values(by=["numeric_isbn"], ascending=True)
print(sorted_books_df.tail(20))
print("----------------------------------------------")

#Processing the isbn column in Book-Ratings dataset
strip_ratings_isbn_list = []
for isbn in ratings_df.loc[:,"isbn"]:
    strip_ratings_isbn = isbn.rstrip(isbn[-1])
    strip_ratings_isbn_list.append(strip_ratings_isbn)

ratings_df.insert(2, "stripped_isbn", strip_ratings_isbn_list)
ratings_df = ratings_df[ratings_df["stripped_isbn"].astype(str).str.isnumeric() == True]

num_ratings_isbn_list = []
for isbn in ratings_df.loc[:,"stripped_isbn"]:
    num_ratings_isbn_list.append(int(isbn))

ratings_df.insert(3, "numeric_isbn", num_ratings_isbn_list)
ratings_df = ratings_df.drop(columns=["isbn","stripped_isbn"])
sorted_ratings_df1 = ratings_df.sort_values(by=["numeric_isbn"], ascending=True)
print(sorted_ratings_df1.tail(20))
print("----------------------------------------------")

"""
Step 5: Convert user_id to numeric numbers in the correct order
"""
print("========== Step 5: Convert user_id to numeric numbers in the correct order ==========")
num_user_id_list = []
for id in users_df["user_id"]:
    id = int(id)
    num_user_id_list.append(id)

users_df = users_df.drop(columns=["user_id"])
users_df["user_id"] = num_user_id_list
users_df = users_df[["user_id","Location","Age"]]
sorted_users_df = users_df.sort_values(by=["user_id"], ascending=True)
print(sorted_users_df.head(20))
print("----------------------------------------------")

ratings_user_id_list = []
for id in ratings_df["user_id"]:
    id = int(id)
    ratings_user_id_list.append(id)

ratings_df = ratings_df.drop(columns=["user_id"])
ratings_df["user_id"] = ratings_user_id_list
ratings_df = ratings_df[["user_id","numeric_isbn","rating"]]
sorted_ratings_df2 = ratings_df.sort_values(by=["user_id"], ascending=True)
print(sorted_ratings_df2.head(20))
print("----------------------------------------------")

"""
Step 6: Convert both user_id and ISBN to the ordered list
"""
print("========== Step 6: Convert both user_id and ISBN to the ordered list ==========")
#In books_df
sorted_isbn = sorted(num_isbn_list)
sorted_unique_isbn = sorted(set(sorted_isbn))
print("Total unique isbn in books_df: ",len(sorted_unique_isbn))
print(sorted_unique_isbn[:100])

#In ratings_df
sorted_ratings_isbn = sorted(num_ratings_isbn_list)
sorted_unique_ratings_isbn = sorted(set(sorted_ratings_isbn))
print("Total unique isbn in ratings_df: ",len(sorted_unique_ratings_isbn))
print(sorted_unique_ratings_isbn[:100])

#In users_df
sorted_user_id = sorted(num_user_id_list)
sorted_unique_user_id = sorted(set(sorted_user_id))
print("Total unique users in users_df: ",len(sorted_unique_user_id))
print(sorted_unique_user_id[:100])

#In ratings_df
sorted_ratings_user_id = sorted(ratings_user_id_list)
sorted_unique_ratings_user_id = sorted(set(sorted_ratings_user_id))
print("Total unique users who had given ratings: ",len(sorted_unique_ratings_user_id))
print(sorted_unique_ratings_user_id[:100])

"""
Step 7: Re-index the columns to build a matrix

In this step we need to merge the columns from different datasets to form the ultimate dataframe for building the ML model.
I will first move all the columns except isbn from Books dataframe to the Book-Ratings dataframe.
Then I will move Location and Age column from the Users dataframe to the Book-Ratings dataframe.
Finally at this point we could have an ultimate dataframe which contains all users who had given ratings with
all neccessary features of users themselves and the books they had rated for.

sorted_ratings_df1 = sorted by numeric_isbn
sorted_ratings_df2 = sorted by user_id
"""
print("========== Step 7: Re-index the columns to build a matrix ==========")
#Merging dataframes to build the initial version of the final df
books_ratings_merged_df = sorted_ratings_df1.merge(sorted_books_df)
sorted_merged_df = books_ratings_merged_df.sort_values(by=["user_id"], ascending=True)
ultimate_df = sorted_merged_df.merge(sorted_users_df)
ultimate_df.drop(columns=["book_title"])

#Encoding the location, author and publisher data into numerical variable
location_str_list = []
for location in ultimate_df.loc[:,"Location"]:
    locstr = location.split(',')
    location_str_list.append(locstr)

user_country_list = [i[2] for i in location_str_list]
ultimate_df.insert(2, "Country", user_country_list)
ultimate_df = ultimate_df.dropna(subset=["Country","book_author","publisher"])

selected_column = ["Country", "book_author", "publisher"]
for column in selected_column:
    ultimate_df[column] = ultimate_df[column].astype("category")
    ultimate_df[column+"_Cat"] = ultimate_df[column].cat.codes

country_unique_list = ultimate_df["Country"].value_counts().index.tolist()
author_unique_list = ultimate_df["book_author"].value_counts().index.tolist()
publisher_unique_list = ultimate_df["publisher"].value_counts().index.tolist()
ultimate_df.drop(columns=["Location","Country","book_author","publisher"])

#One extra step to open a new column named "recommend", true if the ratings given by users is >= 8
recommend_boolean_list = ["True" if r >= 8 else "False" for r in ultimate_df["rating"]]
ultimate_df.insert(len(ultimate_df.columns), "recommend", recommend_boolean_list)

#Re-arrange the columns of the ultimate df
ultimate_df = ultimate_df[["user_id","Country_Cat","Age","numeric_isbn","book_author_Cat","pub_year","publisher_Cat","rating","recommend"]]

#Print out the dtype of the column of df to ensure all of them are numerical variables
df_dtype_list = ultimate_df.dtypes
print(df_dtype_list)

print(ultimate_df.head(20))  #Print out the first 20 columns of the ultimate df to check if everything is alright.
ultimate_df.to_csv("ultimate_df.csv",index=False) #saving the ultimate df to local disk as csv file.

"""
Step 8: Split the data into Training and Testing sets
"""
print("========== Step 8: Split the data into Training and Testing sets ==========")
#In case of too long running time when testing the ml models, I select only the first 100k rows here
ultimate_df = ultimate_df[:100000]

#For training of prediction of ratings, which is a regression task
X = ultimate_df.drop(columns=["rating","recommend"])
y = ultimate_df["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X.head(20))

"""
Step 9: Build a ML model to make ratings predictions based on user and item variables
"""
print("========== Step 9: Build a ML model to Make predictions based on user and item variables ==========")
cv_folds = 5
scoring = "neg_mean_absolute_error"

reg_models = []
reg_models.append(('LR', LinearRegression()))
reg_models.append(('EN', ElasticNet()))
reg_models.append(('KNN', KNeighborsRegressor()))
reg_models.append(('ABR', AdaBoostRegressor())) # Boosting methods
reg_models.append(('RFR', RandomForestRegressor())) # Bagging methods

reg_names = []
reg_kfold_results = []
reg_train_rmse = []
reg_test_rmse = []

#Create the ML Model
for name, model in reg_models:
    reg_names.append(name)
## k-fold analysis:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=1)
#converted neg mean absolute error to positive, i.e. The lower the better
    cv_results = -1 * cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    reg_kfold_results.append(cv_results)
# Full Training period
    res = model.fit(X_train, y_train)
    train_result_mse = mean_squared_error(res.predict(X_train), y_train)
    train_result_rmse = sqrt(train_result_mse)
    reg_train_rmse.append(train_result_rmse)
# Test results
    test_result_mse = mean_squared_error(res.predict(X_test), y_test)
    test_result_rmse = sqrt(test_result_mse)
    reg_test_rmse.append(test_result_rmse)
#Print out the results of each tested model
    msg = "%s: %f (%f) %f %f" % (name, cv_results.mean(), cv_results.std(), train_result_rmse, test_result_rmse)
    print(msg)

"""
Step 10: ML Model Evaluation and Finalising the Model
"""
print("========== Step 10: ML Model Evaluation and Finalising the Model ==========")
#Compare regression algorithms
kfold_fig = plt.figure()
kfold_fig.suptitle('Regression Algorithm Comparison: Kfold results')
ax = kfold_fig.add_subplot(111)
plt.boxplot(reg_kfold_results)
ax.set_xticklabels(reg_names)
kfold_fig.set_size_inches(15,8)
plt.show()

traintesterr_fig = plt.figure()
ind = np.arange(len(reg_names)) # the x locations for the groups
width = 0.35 # the width of the bars
traintesterr_fig.suptitle('Regression Algorithm Comparison: Train vs Test RMSE')
ax = traintesterr_fig.add_subplot(111)
plt.bar(ind - width/2, reg_train_rmse, width=width, label='Train Error')
plt.bar(ind + width/2, reg_test_rmse, width=width, label='Test Error')
traintesterr_fig.set_size_inches(15,8)
plt.legend()
ax.set_xticks(ind)
ax.set_xticklabels(reg_names)
plt.show()

#Model Tuning and Grid Search for Linear Regression model
lin_param_grid = {'fit_intercept': [True, False]}
lin_model = LinearRegression()
lin_kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=1)
lin_grid = GridSearchCV(estimator=lin_model, param_grid=lin_param_grid, scoring=scoring, cv=lin_kfold)
lin_grid_result = lin_grid.fit(X_train, y_train)
print("Result of Model Tuning and Grid Searching for Linear Regression Model")
print("Best: %f using %s" % (lin_grid_result.best_score_, lin_grid_result.best_params_))
lin_means = lin_grid_result.cv_results_['mean_test_score']
lin_stds = lin_grid_result.cv_results_['std_test_score']
lin_params = lin_grid_result.cv_results_['params']
for mean, stdev, param in zip(lin_means, lin_stds, lin_params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("--------------------------------------------------------")

#Model Tuning and Grid Search for Elastic Net Model
en_param_grid = {'alpha': [0.01, 0.1, 0.5, 1, 2, 3],'l1_ratio': [0.01, 0.1, 0.5, 0.9, 0.99]}
en_model = ElasticNet()
en_kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=1)
en_grid = GridSearchCV(estimator=en_model, param_grid=en_param_grid, scoring=scoring, cv=en_kfold)
en_grid_result = en_grid.fit(X_train, y_train)
print("Result of Model Tuning and Grid Searching for Elastic Net Model")
print("Best: %f using %s" % (en_grid_result.best_score_, en_grid_result.best_params_))
en_means = en_grid_result.cv_results_['mean_test_score']
en_stds = en_grid_result.cv_results_['std_test_score']
en_params = en_grid_result.cv_results_['params']
for mean, stdev, param in zip(en_means, en_stds, en_params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("--------------------------------------------------------")

#Model Tuning and Grid Search for AdaBoosting Model
ada_param_grid = {'n_estimators': [50,100,200,300,400],'learning_rate': [1, 2, 3]}
ada_model = AdaBoostRegressor(random_state=1)
ada_kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=1)
ada_grid = GridSearchCV(estimator=ada_model, param_grid=ada_param_grid, scoring=scoring, cv=ada_kfold)
ada_grid_result = ada_grid.fit(X_train, y_train)
print("Result of Model Tuning and Grid Searching for AdaBoosting Model")
print("Best: %f using %s" % (ada_grid_result.best_score_, ada_grid_result.best_params_))
ada_means = ada_grid_result.cv_results_['mean_test_score']
ada_stds = ada_grid_result.cv_results_['std_test_score']
ada_params = ada_grid_result.cv_results_['params']
for mean, stdev, param in zip(ada_means, ada_stds, ada_params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("--------------------------------------------------------")

#Model Tuning and Grid Search for Random Forest Model
rf_param_grid = {'n_estimators': [50,100,200,300,400]}
rf_model = RandomForestRegressor()
rf_kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=1)
rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, scoring=scoring, cv=rf_kfold)
rf_grid_result = rf_grid.fit(X_train, y_train)
print("Result of Model Tuning and Grid Searching for Random Forest Model")
print("Best: %f using %s" % (rf_grid_result.best_score_, rf_grid_result.best_params_))
rf_means = rf_grid_result.cv_results_['mean_test_score']
rf_stds = rf_grid_result.cv_results_['std_test_score']
rf_params = rf_grid_result.cv_results_['params']
for mean, stdev, param in zip(rf_means, rf_stds, rf_params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("--------------------------------------------------------")

#Finalise the model, random forest is showing the best results so I will choose Random Forest as the final model
model = RandomForestRegressor(n_estimators=400) #best params for rf model is 400
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("RMSE for the final model is:", sqrt(mean_squared_error(y_test, predictions)))
print("R2 Score for the final model is:", r2_score(y_test, predictions))
print(model.feature_importances_)  #use inbuilt class feature_importances of tree based regressors
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

#Save Model for Later Use
filename = 'finalized_model.sav'
dump(model, open(filename, 'wb'))