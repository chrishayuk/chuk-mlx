# test_scheduler.py
import math
import pytest
from training.training_scheduler import schedule_learning_rate

class MockOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

# Assume schedule_learning_rate is imported from the module where it's defined
# from your_module import schedule_learning_rate

@pytest.fixture
def optimizer():
    return MockOptimizer(learning_rate=0.1)

def test_warmup_scheduler(optimizer):
    initial_lr = 0.1
    optimizer.learning_rate = initial_lr
    warmup_steps = 10

    lr = schedule_learning_rate(optimizer, 0, warmup_steps, scheduler_type='warmup')
    expected_lr = initial_lr / warmup_steps
    assert lr == pytest.approx(expected_lr, rel=1e-2), "Failed at iteration 0"

    lr = schedule_learning_rate(optimizer, warmup_steps - 1, warmup_steps, scheduler_type='warmup')
    expected_lr = initial_lr
    assert lr == pytest.approx(expected_lr, rel=1e-2), "Failed at iteration warmup_steps - 1"


def test_linear_decay_scheduler(optimizer):
    initial_lr = 0.1
    optimizer.learning_rate = initial_lr
    total_steps = 100
    min_lr = 0.01

    lr = schedule_learning_rate(optimizer, 0, 0, scheduler_type='linear_decay', total_steps=total_steps, min_lr=min_lr)
    expected_lr = initial_lr
    assert lr == pytest.approx(expected_lr, rel=1e-3), "Failed at iteration 0"

    lr = schedule_learning_rate(optimizer, total_steps - 1, 0, scheduler_type='linear_decay', total_steps=total_steps, min_lr=min_lr)
    expected_lr = min_lr
    assert lr == pytest.approx(expected_lr, rel=1e-3), "Failed at iteration total_steps - 1"

def test_exponential_decay_scheduler(optimizer):
    initial_lr = 0.1
    optimizer.learning_rate = initial_lr
    decay_rate = 0.96
    decay_steps = 10

    lr = schedule_learning_rate(optimizer, 0, 0, scheduler_type='exponential_decay', decay_rate=decay_rate, decay_steps=decay_steps)
    expected_lr = initial_lr
    assert lr == pytest.approx(expected_lr, rel=1e-3), "Failed at iteration 0"

    lr = schedule_learning_rate(optimizer, decay_steps, 0, scheduler_type='exponential_decay', decay_rate=decay_rate, decay_steps=decay_steps)
    expected_lr = initial_lr * (decay_rate ** 1)
    assert lr == pytest.approx(expected_lr, rel=1e-3), "Failed at iteration decay_steps"

def test_cosine_annealing_scheduler(optimizer):
    initial_lr = 0.1
    optimizer.learning_rate = initial_lr
    total_steps = 50
    min_lr = 0.01

    lr = schedule_learning_rate(optimizer, 0, 0, scheduler_type='cosine_annealing', total_steps=total_steps, min_lr=min_lr)
    expected_lr = initial_lr
    assert lr == pytest.approx(expected_lr, rel=1e-3), "Failed at iteration 0"

    lr = schedule_learning_rate(optimizer, total_steps, 0, scheduler_type='cosine_annealing', total_steps=total_steps, min_lr=min_lr)
    expected_lr = min_lr
    assert lr == pytest.approx(expected_lr, rel=1e-3), "Failed at iteration total_steps"
