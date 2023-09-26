from json import load


def avg(lst):
    return round(sum(lst)/len(lst), 4)


baseline = ["m0", "m1", "m2"]
tests = ["test0", "test1"]


res_files = ["data/results.json"]

for res_file in res_files:
    with open(res_file, "r") as jf:
        results = load(jf)

    for model in results:
        results[model] = {x: avg(results[model][x]) for x in tests if x in results[model]}

    for test in tests:
        print(test)
        base_max = max([results[b][test] for b in baseline])
        base_err = 1-base_max

        acc1 = results["perceptron"][test]
        acc2 = results["full"][test]

        print("perceptron acc increase: ", round(acc1/base_max-1, 4))
        print("perceptron err decrease: ", round(1-(1-acc1)/base_err, 4))
        print("full acc increase: ", round(acc2/base_max-1, 4))
        print("full err decrease: ", round(1-(1-acc2)/base_err, 4))
