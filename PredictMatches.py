import pandas as pd

matches = pd.read_csv("matches.csv", index_col=0)
matches.head()
matches.shape

matches.dtypes
## machine learning model can only process float/int types 
matches["date"] = pd.to_datetime(matches["date"]) # converting date to a datetime object
matches["home_away"]  = matches["venue"].astype('category').cat.codes # converting venue to a home (1) or away (0) int numeber
matches["opp_code"] = matches["opponent"].astype("category").cat.codes # giving each opponent a different int number
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int") # converting time to a particular hour
matches["day_code"] = matches["date"].dt.day_of_week # converting each day of the week to a corresponding number

matches["target"] = (matches["result"] == "W").astype("int") # win is given a value of 1; while lose and draw are 0

matches.dtypes # now all the predictors and target we need can be processed by the model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
# higher n_estimator, longer training time but more accurate
# higher min_samples_split, less likely to overfit but lower accuracy on training data

train = matches[matches["date"] < '2022-01-01'] # all matches before 2022
test = matches[matches["date"] > '2022-01-01']
predictors = ["home_away", "opp_code", "hour", "day_code"]

rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])

from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"], preds)
print("Baseline accuracy:", acc) # 61.23%
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])

from sklearn.metrics import precision_score
print("Baseline precision:", precision_score(test["target"], preds)) # 47.46%

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Arsenal")

def rolling_average(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed="left").mean() # makes prediction for the 4th game based on result of the previous 3 games
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    # group["team"] = group.name
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols] # creating new columns with rolling average values 
rolling_average(group, cols, new_cols)

cols_no_team = matches.columns.difference(["team"]) # Exclude the grouping column during apply (silences the warning)

matches_rolling = (
    matches.groupby("team", group_keys=True)[cols_no_team]
           .apply(lambda x: rolling_average(x, cols, new_cols))
           .reset_index(level=0)       # turns "team" at index level 0 into a normal column
           .reset_index(drop=True)     # index numbers now match the number of rows
)
matches_rolling

def make_predictions(data, predictors):
    train = data[data["date"] < "2022-01-01"]
    test = data[data["date"] > "2022-01-01"]
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
print("Precision with rolling stats and performance:", precision) # 62.50% (increased from 47.46%)
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
combined


class MissingDict(dict): # Missing Dict is a subclass of Dict
    __missing__ = lambda self, key: key # returns the key itself if it’s a missing key (key is not found)

map_values = {
    "Brighton and Hove Albion" : "Brighton",
    "Manchester United" : "Manchester Utd",
    "Newcastle United" : "Newcastle Utd", 
    "Tottenham Hotspur" : "Tottenham", 
    'West Ham United' : "West Ham",
    "Wolverhampton Wanderers" : "Wolves"
}
mapping = MissingDict(**map_values)
mapping["Wolverhampton Wanderers"] # now teams like Wolverhampton Wanderers will always be called Wolves

combined["new_team"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
merged

## number of times team X was predicted to win AND team Y was predicted to lose, and actual was win for X
print(merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()) # returns 27/40 = 67.50%



## project inspired by dataquest tutorial on youtube