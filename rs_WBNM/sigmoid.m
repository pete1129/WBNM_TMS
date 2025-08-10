function S = sigmoid(v)
e0 = 2.5;
r = 0.56;
v0 = 6;
S = 2*e0./(1+exp(r*(v0-v)));
end