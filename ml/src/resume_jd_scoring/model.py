from torch import Tensor, nn


class ResumeJDLinearClassifier(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        class_count: int = 3,
    ) -> None:
        super().__init__()
        if input_dimension <= 0:
            raise ValueError("input dimension must be positive")
        if class_count <= 1:
            raise ValueError("class count must exceed one")
        self.projection = nn.Linear(input_dimension, class_count)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)
