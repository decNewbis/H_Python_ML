import torch


if __name__ == "__main__":
    # Computation Graphs and Automatic Differentiation
    # If ``requires_grad=True``, the Tensor object keeps track 
    # of how it was created. Let’s see it in action.

    # Tensor factory methods have a ``requires_grad`` flag
    # Here are X's
    x = torch.tensor([1., 2., 3], requires_grad=True)
    # Here is a funciton. Let's find derivative value
    # for the function in each of X's
    z = 2 * x**2 + 10
    print(f"{z = }")

    # BUT z knows something extra.
    print(f"{z.grad_fn = }")

    # ``s`` is formulated as follows:
    # (1) s = z0 + z1 + z2
    # (2) s = (2 * x0**2 + 10) + (2 * x1**2 + 10) + (2 * x2**2 + 10)
    s = z.sum()

    # So now, what is the derivative of this sum
    # with respect to the first component of x? 
    # In math, we want ``ds / dx0``
    #
    # So having (2):
    # (3) ds / dx0 = 4 * x0
    #
    # For x0 = 1:
    # (4) ds / dx0 = 4 * x0 = 4 * 1 = 4
    s.backward()

    print(f"{s = }")
    print(f"{x.grad = }")
