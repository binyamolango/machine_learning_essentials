### Dropout Example


For each element, we either drop it to `0` or keep it and scale it by `1 / (1 - p_drop)`.

This small example shows the idea on a tiny tensor.

```
torch.manual_seed(0)


X = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

p_drop = 0.5


Y = dropout(X, p_drop=p_drop)


print("Input:", X)


print("Output:", Y)
```

drop_mask tensor([[1., 0., 1., 1.]])

keep_mask tensor([[0., 1., 0., 0.]])

kept_X tensor([[0., 2., 0., 0.]])

Input: tensor([[1., 2., 3., 4.]])

Output: tensor([[0., 4., 0., 0.]])