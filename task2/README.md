# Task 2
Used a binary tree splitting on the user id.
Could be improved upon depending on use case.
I didn't write a separate duplicate search, but just recognized duplicates when adding the entry to the tree.
I associated a purchase entry with a play time of 0.
The duplication logic is found [here](./game_tree.f90#L46).
It will fail in the following case and return an incorrect play time:

1. An entry is added with play time A
2. An entry is added with play time B
3. A (duplicate) entry with play time A or B is added

This could easily be fixed by just storing all the unique play times in an array instead of adding them.

## Compile
```
gfortran -o game_tree game_tree.f90
```

## Run
```
./game_tree ../data/algorithms\ part\ dataset.csv
```
