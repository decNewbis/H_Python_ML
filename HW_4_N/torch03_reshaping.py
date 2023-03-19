import torch


if __name__ == "__main__":
    # Reshaping Tensors
    x = torch.randn(2, 3, 4)
    print(x)
    # Reshape to 2 rows, 12 columns
    print(x.view(2, 12))
    # Same as above. 
    # If one of the dimensions is -1,
    # its size can be inferred
    print(x.view(2, -1))
