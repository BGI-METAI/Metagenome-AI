from collections import Counter
import csv
import sys
import itertools

from sklearn.model_selection import train_test_split

MAX_SEQ_LENGTH = 3000
MIN_PFAM_CNT = 100


def describe_dataset(dataset, desc):
    """Just for basic info e.g. number of families, min count etc."""
    print(desc)
    cnt = Counter([s[3] for s in dataset])
    print(f"There is a total of {len(cnt)} different pfams in the first place")
    print(f"Minimum count of family in the first place is {min(cnt.values())}")
    cnt = Counter(list(itertools.chain.from_iterable([s[3:] for s in dataset])))
    print(f"There is a total of {len(cnt)} different pfams overall")
    print(f"Minimum count of family overall is {min(cnt.values())}")
    print(" ")


def get_low_count_families(dataset):
    """Get families that are less abundant than *some value*"""
    cnt = Counter(list(itertools.chain.from_iterable([s[3:] for s in dataset])))
    to_remove = set([k for k, v in cnt.items() if v < 10])
    print(f"To be removed: {len(to_remove)} families")
    return to_remove


def filter_datasets(datasets):
    to_remove = set()
    for dataset in datasets:
        to_remove = to_remove.union(get_low_count_families(dataset))
    cnt_remove = len(to_remove)
    print(f"There are {cnt_remove} families to be removed from all datasets")
    for dataset in datasets:
        dataset[:] = [
            sample
            for sample in dataset
            if len(set(sample[3:]).intersection(to_remove)) == 0
        ]
    print("Removed some samples...")
    return cnt_remove


def check_family_presence(train, test, val):
    train = set(list(itertools.chain.from_iterable([s[3:] for s in train])))
    print(f"There is a total of {len(train)} different pfams in TRAIN set")
    test = set(list(itertools.chain.from_iterable([s[3:] for s in test])))
    print(f"There is a total of {len(test)} different pfams in TEST set")
    val = set(list(itertools.chain.from_iterable([s[3:] for s in val])))
    print(f"There is a total of {len(val)} different pfams in VAL set")

    cnt_val_in_train = len(val.intersection(train))
    cnt_test_in_train = len(test.intersection(train))
    print(f"Intersection of VAL and TRAIN: {cnt_val_in_train}")
    print(f"Intersection of TEST and TRAIN: {cnt_test_in_train}")


path = sys.argv[1]

with open(path) as file:
    reader = csv.reader(file, delimiter=" ")
    samples = list(reader)

# filter for length
samples = [sample for sample in samples if int(sample[1]) < MAX_SEQ_LENGTH]
print(f"Filtered for length less than {MAX_SEQ_LENGTH}")

# filter for families
print(
    f"In first place there is {len(set([s[3] for s in samples]))} different pfamilies\n"
)
cnt = Counter([s[3] for s in samples])
cnt_pfam_before = len(cnt)
taken_pfam = [k for k, v in cnt.items() if v > MIN_PFAM_CNT]
taken_pfam = set(taken_pfam)
samples = [sample for sample in samples if sample[3] in taken_pfam]
print(
    f"Filtered for families more abundant than {MIN_PFAM_CNT}. Choosen {len(taken_pfam)} out of {cnt_pfam_before}\n"
)

X = [sample for sample in samples]
y = [sample[3] for sample in samples]

print(
    f"In the first place sample[3:], there are a total of {len(set(y))} different pfams\n"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
)
describe_dataset(X_train, "Describing TRAIN set")
describe_dataset(X_test, "Describing TEST set")
describe_dataset(X_val, "Describing VALIDATION set")

while filter_datasets([X_train, X_test, X_val]) > 0:
    # do nothing, wait until it removes needed samples
    pass

describe_dataset(X_train, "Describing TRAIN set after filtering")
describe_dataset(X_test, "Describing TEST set after filtering")
describe_dataset(X_val, "Describing VALIDATION set after filtering")

check_family_presence(X_train, X_test, X_val)

with open("pfam_train.csv", "w") as train, open("pfam_test.csv", "w") as test, open(
    "pfam_validation.csv", "w"
) as val:
    writer = csv.writer(train, delimiter=" ")
    writer.writerows(X_train)
    writer = csv.writer(test, delimiter=" ")
    writer.writerows(X_test)
    writer = csv.writer(val, delimiter=" ")
    writer.writerows(X_val)

print("Finished!")
