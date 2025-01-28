include "stable_matching.mzn";
include "minicpp.mzn";

int: n = 5;

var 0..n-1: m0;
var 0..n-1: m1;
var 0..n-1: m2;
var 0..n-1: m3;
var 0..n-1: m4;

var 0..n-1: w0;
var 0..n-1: w1;
var 0..n-1: w2;
var 0..n-1: w3;
var 0..n-1: w4;

array [int, int] of 0..n-1: pm = [|
    3, 4, 0, 1, 2,
    4, 2, 0, 3, 1,
    3, 4, 1, 0, 2,
    0, 4, 3, 1, 2,
    2, 3, 1, 4, 0
|];

array [int, int] of 0..n-1: pw = [|
    3, 2, 0, 1, 4,
    1, 4, 0, 3, 2,
    1, 2, 4, 0, 3,
    3, 1, 4, 0, 2,
    1, 2, 0, 3, 4
|];

constraint m0 = 3;
constraint m1 = 2;
constraint m2 < 3;
constraint m3 != 1;
constraint m3 != 2;
constraint m3 != 3;
constraint m4 > 1;
constraint m4 < 4;

% Constraints for women's domains based on the domain matrix
constraint w0 != 0;
constraint w0 != 2;
constraint w0 != 4;
constraint w1 > 1;
constraint w1 < 4;
constraint w2 != 0;
constraint w2 != 3;
constraint w3 < 2;
constraint w4 != 1;
constraint w4 < 3;

% Stability constraints
constraint stable_matching([m0, m1, m2, m3, m4], [w0, w1, w2, w3, w4], pm, pw) ::uniud;

solve satisfy;
