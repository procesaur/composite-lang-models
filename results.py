from json import load
import numpy as np
from scipy.stats import t as t_dist


def avg(lst):
    return round(sum(lst)/len(lst), 4)


def paired_t_test_old(p):
    mean = np.mean(p)
    n = len(p)
    den = np.sqrt(sum([(diff - mean) ** 2 for diff in p]) / (n - 1))
    t = (mean * (n ** (1 / 2))) / den
    p_value = t_dist.sf(t, n - 1) * 2
    return t, p_value


def paired_t_test(diff, n, ratio):
    n1 = round(n*ratio)
    n2 = n-n1
    diff = np.array(diff)
    mean = np.mean(diff)
    sigma2 = np.var(diff)
    mod = 1/len(diff) + n2/n1
    sigma2_mod = sigma2 * mod
    tt = mean/np.sqrt(sigma2_mod)
    pval = t_dist.sf(tt, n - 1) * 2
    return tt, pval


baseline = ["m0", "m1", "m2"]
tests = ["test0", "test1"]
samples = 52236
ratio = 0.2

res_file = "data/results.json"
duplicates_file = "data/duplicates.json"


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


def results2():
    with open(res_file, "r") as jf:
        results = load(jf)

    with open(duplicates_file, "r") as jf:
        duplicates = load(jf)

    for i, test in enumerate(tests):
        print(test)
        n_duplicates = len(duplicates[str(i)])

        perceptron_inc = []
        full_inc = []

        for i in range(5):
            base_max = max([results[b][test][i] for b in baseline])
            perceptron_inc.append(results["perceptron"][test][i]-base_max)
            full_inc.append(results["full"][test][i]-base_max)

        t, p = paired_t_test(perceptron_inc, samples-n_duplicates, ratio)
        print(f"perceptron t statistic: {round(t,4)}, p-value: {round(p,4)}")
        t, p = paired_t_test(full_inc, samples-n_duplicates, ratio)
        print(f"full t statistic: {round(t,4)}, p-value: {round(p,4)}")


results()
results1()
results2()
