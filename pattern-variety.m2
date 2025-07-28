restart
-*
This code computes the generators in the ideal of the pattern variety
for a given neural network. Need to specify right away:

. L (the number of layers - 1)
. n_1, ..., n_L (the widths of the layers)
. P (the activation patterns, listed by block)

Moreover, specify the degree up to which the invariants 
should be computed later in the code.
*-


------------------------------------------------------------------------------------

-- example: shallow net (Ex. 37)

L = 2

n_0 = 3
n_1 = 4
n_2 = 3

P = {{{1, 1, 0, 0}}, {{1, 0, 1, 0}}, {{1, 0, 0, 1}}}; -- 3 blocks


------------------------------------------------------------------------------------

-- example: deep net (Ex. 31)

restart

L = 4

n_0 = 3
n_1 = 3
n_2 = 2
n_3 = 3
n_4 = 3


P = {{{1, 1, 1}, {1, 1}, {1, 1, 1}}, {{0, 1, 1}, {1, 1}, {1, 1, 1}}} -- 2 blocks


------------------------------------------------------------------------------------


--------
-- setup
--------
ws  = flatten apply(splice([1..L]), i -> toList(w_(i,(1,1))..w_(i,(n_i,n_(i-1))))) 
var = toList join(ws)


S = QQ[var]
R = QQ[y_1..y_(n_0*n_L*(length P))]

for i from 1 to L do {
    W_i = transpose genericMatrix(S, w_(i,(1,1)), n_(i-1), n_i);
    };

Ms = {}; for A in P do {
Ap = apply(A, el -> diagonalMatrix(el)); Ap = append(Ap, id_(S^(n_(L)))); Ap = apply(Ap, a -> promote(a, S));

M = id_(S^(n_0)); for el in apply(1..L, i -> Ap_(i-1) * W_i) do { M = el * M; };
Ms = append(Ms, M);
};

Ms = unique(Ms); bigM = Ms_0; 
for i from 1 to (length P) - 1 do bigM = bigM | Ms_i;


param = flatten entries(transpose bigM);


---------------------------------------------
--- compute invariants up to a certain degree
---------------------------------------------
needsPackage "MultigradedImplicitization"
phi = map(S, R, param);
G = componentsOfKernel(3, phi); -- SPECIFY DEGREE HERE
G = apply(flatten values G, el -> substitute(el, R));
JM = trim ideal(G);
betti mingens JM
netList JM_*
dim JM


----------------------------------
--- compute dimension via jacobian
----------------------------------
jac = jacobian(ideal(param));
subz = apply(gens(S), w -> w => random(1,50))
j = sub(jac, subz);
rank j



dim JM == rank j -- if true, no other generators are needed










