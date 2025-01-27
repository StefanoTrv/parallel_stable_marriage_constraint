include "stable_matching.mzn";
include "minicpp.mzn";

int: n = 3;

var 0..n-1: m0;
var 0..n-1: m1;
var 0..n-1: m2;

var 0..n-1: w0;
var 0..n-1: w1;
var 0..n-1: w2;

array [int, int] of 0..n-1: pm = [|0,1,2|2,0,1|2,0,1|];
array [int, int] of 0..n-1: pw = [|2,0,1|2,1,0|0,2,1|];

% Directly enforce domain constraints for women
constraint w2 != 2;  % Woman 2 cannot be paired with man 2

constraint stable_matching([m0, m1, m2], [w0, w1, w2], pm, pw) ::uniud;

solve satisfy;
