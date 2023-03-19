##%%
import torch


if __name__ == "__main__":
    torch.manual_seed(1)

    # Creating Tensors
    # Tensors can be created from Python lists
    # with the torch.tensor() function.
    #
    # torch.tensor(data) creates a torch.Tensor object 
    # with the given data.
    V_data = [1., 2., 3.]
    V = torch.tensor(V_data)
    print(V)

    # Creates a matrix
    M_data = [[1., 2., 3.], [4., 5., 6]]
    M = torch.tensor(M_data)
    print(f'{M = }')

    # Create a 3D tensor of size 2x2x2.
    T_data = [[[1., 2.], [3., 4.]],
              [[5., 6.], [7., 8.]]]
    T = torch.tensor(T_data)
    print(T)

    # -------------------------------------

    # # Index into V and get a scalar (0 dimensional tensor)
    # print(V[0])
    # # Get a Python number from it
    # print(V[0].item())

    # # Index into M and get a vector
    # print(M[0])

    # # Index into T and get a matrix
    # print(T[0])
    
    # -------------------------------------

#     # Math operations
#     a = torch.tensor([1, 2, 3])
#     b = torch.tensor([4, 5, 6])
#     const = 10

#     print(f"{a + b = }")
#     print(f"{a - b = }")
#     print(f"{a / b = }")
#     print(f"{a * b = }")
#     print(f"{a ** b = }")

#     # Is it possible?
#     print(f"{a * b + const = }")

#     # Dot product
#     print(f"{a @ b = }")
#     print(f"{a.matmul(b) = }")

#     # Matrix multiplication
#     A = torch.tensor([
#         [1, 2, 3],
#         [1, 2, 3],
#     ])
#     B = torch.tensor([
#         [1, 2],
#         [3, 4],
#         [5, 6],
#     ])
#     print(f"{A.matmul(B) = }")
#     # Is it possible? HW
#     print(f"{B.matmul(A) = }")

    # -------------------------------------

#     # You can create a tensor with random data 
#     # and the supplied dimensionality with torch.randn()
    x = torch.randn((3, 4, 5))
    print(x)
