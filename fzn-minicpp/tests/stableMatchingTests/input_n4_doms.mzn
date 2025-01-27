include "stable_matching.mzn";
include "minicpp.mzn";

int: n = 4;

var 0..n-1: m0;
var 0..n-1: m1;
var 0..n-1: m2;
var 0..n-1: m3;

var 0..n-1: w0;
var 0..n-1: w1;
var 0..n-1: w2;
var 0..n-1: w3;

array [int, int] of 0..n-1: pm = [|
    2, 1, 3, 0 | 
    1, 0, 2, 3 | 
    3, 0, 2, 1 | 
    0, 2, 1, 3 | 
|];

array [int, int] of 0..n-1: pw = [|
    1, 3, 0, 2 | 
    3, 2, 1, 0 | 
    2, 1, 3, 0 | 
    0, 2, 1, 3 | 
|];

% Domain constraints for men
constraint m0 in {0,1,2,3} diff {0,1};  % Man 0 cannot pair with women 0 or 1
constraint m2 in {0,1,2,3} diff {0,1,2}; % Man 2 cannot pair with women 0, 1, or 2

constraint stable_matching([m0, m1, m2, m3], [w0, w1, w2, w3], pm, pw) ::uniud;

solve satisfy;
