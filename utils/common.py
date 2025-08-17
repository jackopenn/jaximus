

def pretty_log(step, metrics):
    print(f"step: {step}", end=", ")
    for k, v in metrics.items():
        # if isinstance(v, float):
        #     print(f"{k}: {v:.5f}", end=", ")
        # else:
        #     print(f"{k}: {v}", end=", ")
        print(f"{k}: {v:.5f}", end=", ")
    print()