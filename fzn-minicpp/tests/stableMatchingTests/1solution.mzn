include "stable_matching.mzn";
include "minicpp.mzn";

int: n = 3;

var 0..n-1: m0;
var 0..n-1: m1;
var 0..n-1: m2;

var 0..n-1: w0;
var 0..n-1: w1;
var 0..n-1: w2;

array [int, int] of 0..n-1: pm = [|0,1,2|0,1,2|0,1,2|];
array [int, int] of 0..n-1: pw = [|0,1,2|0,1,2|0,1,2|];

constraint stable_matching([m0, m1, m2], [w0, w1, w2], pm, pw) ::uniud;

solve satisfy;

%Variable domain:  0 1 2
%Corresponding solution: 0 1 2
