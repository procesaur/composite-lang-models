from json import load
# import numpy as np


def avg(lst):
    return round(sum(lst)/len(lst), 4)


baseline = ["m0", "m1", "m2"]
tests = ["test0", "test1"]


res_file = "data/results.json"


def results():
    with open(res_file, "r") as jf:
        results = load(jf)

    for test in ["test0", "test1"]:
        for i in range(5):
            rr = ["fold "+str(i+1)]
            for model in results:
                rr.append("{:.4f}".format(round(results[model][test][i], 4)))
            print("&".join(rr)+"\\\\")
        print("\midrule")
        av = ["$\mu$"]
        ma = ["$Max$"]
        for model in results:
            ma.append("{:.4f}".format(round(max(results[model][test]), 4)))
            av.append("{:.4f}".format(round(avg(results[model][test]), 4)))
        print("&".join(ma) + "\\\\")
        print("&".join(av) + "\\\\")


def results1():
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


results()
results1()
