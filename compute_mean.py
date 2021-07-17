import os

base_paths = ["sentiment_results_run1","stance_results_run1"]
for base_path in base_paths:

    print("\n", base_path)
    domains = os.listdir(base_path)
    for domain in domains:
        files = os.listdir(os.path.join(base_path, domain))
        res = [0] * 4
        num_folds = len(files)
        assert num_folds == 5
        for file in files:
            with open(os.path.join(base_path, domain, file)) as f:
                for indx, line in enumerate(f):
                    res[indx] += float(line.split(" ")[-1]) / num_folds

        print(f"{domain}: acc_in: {round(res[0],3)}, f1_in: {round(res[2],3)},"
              f"acc_cross: {round(res[1],3)},  f1_cross: {round(res[3],3)}")
