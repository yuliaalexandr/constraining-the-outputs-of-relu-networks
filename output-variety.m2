restart
-*
This code computes the generators in the ideal of the output variety
for a given neural network with a single block. Need to specify:

. L (the number of layers - 1)
. n_1, ..., n_L (the widths of the layers)
. X (the dataset)
. P (the activation pattern, common for all data)
*-


------------------------------------------------------------------------------------

-- example: shallow net (random dataset)

L = 2

n_0 = 4
n_1 = 3
n_2 = 4

m = 8;
X  = apply(m, j -> vector(apply(n_0, i -> random(1,15))));
P = {{1, 1, 0}}



------------------------------------------------------------------------------------

-- example: deep net (specific dataset)

restart

L = 4

n_0 = 3
n_1 = 3
n_2 = 2
n_3 = 3
n_4 = 3


X = apply({{1, 2, 3}, {5, 6, 11}, {8, 9, 10}, {7, 12, 13}}, el -> vector(el))
m = length X
P = {{1, 1, 1}, {1, 1}, {1, 1, 1}}






------------------------------------------------------------------------------------

ws  = flatten apply(splice([1..L]), i -> toList(w_(i,(1,1))..w_(i,(n_i,n_(i-1))))) 
var = toList join(ws)


S = QQ[y_1..y_(m * n_L), var]


for i from 1 to L do {
    W_i = transpose genericMatrix(S, w_(i,(1,1)), n_(i-1), n_i);
    };

Ms = {}; 
Ap = apply(P, el -> diagonalMatrix(el)); Ap = append(Ap, id_(S^(n_(L)))); Ap = apply(Ap, a -> promote(a, S));

M = id_(S^(n_0)); 

for el in apply(1..L, i -> Ap_(i-1) * W_i) do { M = el * M; };

param = flatten entries(transpose (M * matrix(X)));



-- elimination is usually faster in this case, compared to MultigradedImplicitization.m2
I = ideal(param - (gens S)_{0.. m*n_L - 1});
J = eliminate(I, (gens S)_{m*n_L..(length gens S - 1)});
betti mingens J
netList J_*
dim J


