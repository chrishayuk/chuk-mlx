import mlx.core as mx


class MockOptimizer:
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate

    def update(self, model, gradients):
        # Ensure gradients are numpy arrays and perform update
        for param, grad in zip(model.trainable_parameters, gradients):
            if isinstance(grad, list):
                grad = mx.array(grad)
            param -= self.learning_rate * grad
