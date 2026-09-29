from train import Value

x = Value(3.0)
y = (x + 2) * (x - 1)
y.backward()

print("y =", y.data)
print("x.grad =", x.grad)
print("expected =", 2*x.data + 1)

assert y.data == 10.0, f"forward pass wrong: got {y.data}, expected 10.0"
assert x.grad == 2*x.data + 1, f"backward pass wrong: got {x.grad}, expected {2*x.data + 1}"
print("OK: autograd forward and backward are correct")
