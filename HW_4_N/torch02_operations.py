import torch


if __name__ == "__main__":
    # Operations with Tensors
    x = torch.tensor([1., 2., 3.])
    y = torch.tensor([4., 5., 6.])
    z = x + y
    print(f"{z = }")

    # By default, it concatenates along the first axis 
    # (concatenates rows)
    x_1 = torch.tensor([[ 1,  2], [ 4,  5]])
    y_1 = torch.tensor([[-1, -2], [-4, -5]])
    z_1 = torch.cat([x_1, y_1])
    print(f"{z_1 = }")

    # Concatenate columns:
    x_2 = torch.tensor([[ 1,  2,  3], [ 4,  5,  6]])
    y_2 = torch.tensor([[-1, -2, -3], [-4, -5, -6]])
    # second arg specifies which axis to concat along
    z_2 = torch.cat([x_2, y_2], 1)
    print(f"{z_2 = }")

    # If your tensors are not compatible, torch will complain.  
    # Uncomment to see the error
    # torch.cat([x_1, x_2])