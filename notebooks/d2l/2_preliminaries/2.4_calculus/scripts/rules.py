# Constant Multiplication Rule
# d/dx[c * f(x)] = c * d/dx[f(x)]
# in code
# def constant_multiplication_rule(c, f):
#     return c * f(x)

# Sum Rule
# d/dx[f(x) + g(x)] = d/dx[f(x)] + d/dx[g(x)]
# in code
# def sum_rule(f, g):
#     return f(x) + g(x)

# Product Rule
# d/dx[f(x) * g(x)] = f(x) * d/dx[g(x)] + g(x) * d/dx[f(x)]
# in code
# def product_rule(f, g):
#     return f(x) * g(x)

# Quotient Rule
# d/dx[f(x) / g(x)] = (g(x) * d/dx[f(x)] - f(x) * d/dx[g(x)]) / g(x)^2
# in code
# def quotient_rule(f, g):
#     return (g(x) * d/dx[f(x)] - f(x) * d/dx[g(x)]) / g(x)^2