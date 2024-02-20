import json
import multiprocessing
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Tuple
from warnings import warn

import numpy as np
from tqdm import tqdm

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
# from evalplus.data.utils import CACHE_DIR
from evalplus.eval import (
    # PASS,
    estimate_pass_at_k,
)

PASS = "pass"
from typing import Union, List
import itertools
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
# from evalplus.gen.util import trusted_exec

from evalplus.evaluate import get_groundtruth,check_correctness
# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]


def evaluate_plus(flags):
    if flags.parallel is None:
        n_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        n_workers = flags.parallel

    if os.path.isdir(flags.samples):
        result_path = os.path.join(flags.samples, "eval_results.json")
    else:
        assert flags.samples.endswith(".jsonl")
        result_path = flags.samples.replace(".jsonl", "_eval_results.json")

    if flags.dataset == "humaneval":
        problems = get_human_eval_plus(mini=flags.mini, noextreme=False)
        dataset_hash = get_human_eval_plus_hash(
            mini=flags.mini, noextreme=False
        )
        expected_output = get_groundtruth(problems, dataset_hash, [])
    elif flags.dataset == "mbpp":
        problems = get_mbpp_plus(mini=flags.mini, noextreme=False)
        dataset_hash = get_mbpp_plus_hash(
            mini=flags.mini, noextreme=False
        )
        expected_output = get_groundtruth(
            problems,
            dataset_hash,
            MBPP_OUTPUT_NOT_NONE_TASKS,
        )

    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hash": dataset_hash,
        "eval": {},
    }

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        eval_results = defaultdict(list)  # task_id ->
        remainings = set()

        print("Reading samples...")
        for sample in tqdm(load_solutions(flags.samples)):
            task_id = sample["task_id"]
            solution = (
                sample["solution"]
                if "solution" in sample
                else problems[task_id]["prompt"] + sample["completion"]
            )
            remainings.add(sample["_identifier"])
            args = (
                flags.dataset,
                completion_id[task_id],
                problems[task_id],
                solution,
                expected_output[task_id],
                flags.base_only,
                not flags.test_details,  # fast_check
                sample["_identifier"],
                flags.min_time_limit,
                flags.gt_time_limit_factor,
            )
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        assert n_samples == len(remainings), "Missing problems in unfinished"
        assert len(completion_id) == len(problems), "Missing problems in samples"

        def stucking_checker():
            while remainings:
                last_size = len(remainings)
                time.sleep(20)
                if last_size != len(remainings) or len(remainings) == 0:
                    continue
                # Potential stucking
                warn("No samples had finished testing in the last 20s")
                warn(f"{len(remainings)} samples to be tested: {remainings}")

        threading.Thread(target=stucking_checker).start()

        for future in tqdm(as_completed(futures), total=n_samples):
            result = future.result()
            remainings.remove(result["_identifier"])
            eval_results[result["task_id"]].append(result)

    # sort the results for each problem by completion_id
    for task_id, task_results in eval_results.items():
        task_results.sort(key=lambda x: x["completion_id"])
        results["eval"][task_id] = []
        for res in task_results:

            def get_failed_tests(stat, details, inputs) -> List[Any]:
                if stat == PASS or not details:
                    return []

                if flags.test_details:
                    return [
                        inputs[i] for i in range(len(details)) if not details[i]
                    ]

                # esle => simply return the only and the last fail test
                return [inputs[len(details)]]

            base_stat, base_details = res["base"]
            base_fail_tests = get_failed_tests(
                base_stat, base_details, problems[task_id]["base_input"]
            )

            # initialize plus tests
            plus_stat = None
            plus_fail_tests = []

            # with plus tests
            if not flags.base_only:
                plus_stat, plus_details = res["plus"]
                plus_fail_tests = get_failed_tests(
                    plus_stat, plus_details, problems[task_id]["plus_input"]
                )

            if flags.dataset == "mbpp":
                base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
                plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

            results["eval"][task_id].append(
                {
                    "task_id": task_id,
                    "solution": res["solution"],
                    "base_status": base_stat,
                    "plus_status": plus_stat,
                    "base_fail_tests": base_fail_tests,
                    "plus_fail_tests": plus_fail_tests,
                }
            )

    with open(result_path, "w") as f:
        json.dump(results, f)

    # Calculate pass@k.
    total = np.array([len(r) for r in results["eval"].values()])
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        if not flags.base_only:
            new_correct.append(
                sum(
                    [
                        res[i]["base_status"] == res[i]["plus_status"] == PASS
                        for i in range(len(res))
                    ]
                )
            )
    base_correct = np.array(base_correct)

    pass_at_k = {
        f"pass_{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in [1,2,3,4,5,6,7,8,9, 10, 100]
        if total.min() >= k
    }
    print(f"{flags.dataset} (base tests)")
    for k, v in pass_at_k.items():
        print(f"{k}:\t{v:.3f}")

    if new_correct:
        print(f"{flags.dataset}+ (base + extra tests)")
        pass_at_k_plus = {
            f"pass_{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in [1,2,3,4,5,6,7,8,9, 10, 100]
            if (total >= k).all()
        }
        for k, v in pass_at_k_plus.items():
            print(f"{k}:\t{v:.3f}")
    else:
        pass_at_k_plus = []
    return pass_at_k, pass_at_k_plus    

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",  type=str, choices=["humaneval", "mbpp"], default="humaneval"
    )
    parser.add_argument("--samples",  type=str, default="samples.jsonl")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--parallel", default=None, type=int)
    parser.add_argument("--i-just-wanna-run", action="store_true")
    parser.add_argument("--test-details", action="store_true")
    parser.add_argument("--min-time-limit", default=1, type=float)
    parser.add_argument("--gt-time-limit-factor", default=4.0, type=float)
    parser.add_argument("--mini", action="store_true")

    args = parser.parse_args()

    evaluate_plus(args)

if __name__ == "__main__":
    main()
