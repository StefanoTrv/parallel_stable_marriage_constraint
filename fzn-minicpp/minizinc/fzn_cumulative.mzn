predicate minicpp_cumulative(array[int] of var int: s, array[int] of var int: d, array[int] of var int: r, var int: b);
predicate fzn_cumulative(array[int] of var int: s, array[int] of var int: d, array[int] of var int: r, var int: b) = minicpp_cumulative(s,d,r,b);
