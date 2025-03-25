from datasets import load_dataset


def inspect_dataset():
    dataset = load_dataset("Presto-Design/svg_basic_benchmark_v0")
    eval_data = dataset["test"]

    # Get first example
    first_example = eval_data[0]

    print("Dataset type:", type(eval_data))
    print("\nFirst example type:", type(first_example))
    print("\nFirst example keys:", first_example.keys())
    print("\nImage type:", type(first_example["image"]))

    if hasattr(first_example["image"], "width"):
        print(
            "\nImage dimensions:",
            first_example["image"].width,
            "x",
            first_example["image"].height,
        )


if __name__ == "__main__":
    inspect_dataset()
