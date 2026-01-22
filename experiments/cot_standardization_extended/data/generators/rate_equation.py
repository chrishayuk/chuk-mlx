"""Rate/equation problem generator."""

import random


def generate_rate_time_quantity():
    """Rate × time = quantity."""
    rate = random.randint(5, 50)
    time = random.randint(2, 12)
    result = rate * time

    scenarios = [
        (f"A printer prints {rate} pages per minute. How many pages in {time} minutes?", "pages"),
        (f"A factory makes {rate} widgets per hour. How many in {time} hours?", "widgets"),
        (f"A runner covers {rate} meters per minute. How far in {time} minutes?", "meters"),
        (f"A machine produces {rate} items per day. How many in {time} days?", "items"),
    ]

    question, unit = random.choice(scenarios)

    trace = [
        {"given": {"rate": rate, "unit": f"{unit}/time", "time": time}},
        {"formula": "quantity = rate × time"},
        {"compute": {"op": "mul", "args": [rate, time], "var": "quantity", "result": result}},
        {"query": "quantity"},
    ]

    return {
        "question": question,
        "expert": "rate_equation",
        "trace": trace,
        "answer": result,
    }


def generate_distance_speed_time():
    """Distance = speed × time."""
    speed = random.randint(30, 80)
    time = random.randint(2, 8)
    distance = speed * time

    question = f"A car travels at {speed} km/h. How far does it go in {time} hours?"

    trace = [
        {"given": {"rate": speed, "unit": "km/h", "time": time}},
        {"formula": "distance = speed × time"},
        {"compute": {"op": "mul", "args": [speed, time], "var": "distance", "result": distance}},
        {"query": "distance"},
    ]

    return {
        "question": question,
        "expert": "rate_equation",
        "trace": trace,
        "answer": distance,
    }


def generate_work_rate():
    """Work = rate × time, then divide."""
    rate = random.randint(10, 30)
    time = random.randint(3, 8)
    workers = random.randint(2, 5)

    total_work = rate * time
    per_worker = total_work // workers

    question = f"A team does {rate} tasks per hour. After {time} hours, they split the work among {workers} people. How many tasks per person?"

    trace = [
        {"given": {"rate": rate, "time": time}},
        {"compute": {"op": "mul", "args": [rate, time], "var": "total", "result": total_work}},
        {"compute": {"op": "div", "args": ["total", workers], "var": "per_worker", "result": per_worker}},
        {"query": "per_worker"},
    ]

    return {
        "question": question,
        "expert": "rate_equation",
        "trace": trace,
        "answer": per_worker,
    }


def generate_combined_rate():
    """Two rates combined."""
    rate1 = random.randint(5, 20)
    rate2 = random.randint(5, 20)
    time = random.randint(2, 6)

    combined = rate1 + rate2
    total = combined * time

    question = f"Machine A produces {rate1} items/hour. Machine B produces {rate2} items/hour. How many total in {time} hours?"

    trace = [
        {"given": {"rate_a": rate1, "rate_b": rate2, "time": time}},
        {"compute": {"op": "add", "args": [rate1, rate2], "var": "combined_rate", "result": combined}},
        {"compute": {"op": "mul", "args": ["combined_rate", time], "var": "total", "result": total}},
        {"query": "total"},
    ]

    return {
        "question": question,
        "expert": "rate_equation",
        "trace": trace,
        "answer": total,
    }


GENERATORS = [
    generate_rate_time_quantity,
    generate_distance_speed_time,
    generate_work_rate,
    generate_combined_rate,
]


def generate(n: int = 40) -> list[dict]:
    """Generate n rate equation examples."""
    examples = []
    for _ in range(n):
        gen = random.choice(GENERATORS)
        examples.append(gen())
    return examples


if __name__ == "__main__":
    for ex in generate(3):
        print(f"Q: {ex['question']}")
        print(f"Answer: {ex['answer']}")
        print(f"Trace: {ex['trace']}")
        print()
