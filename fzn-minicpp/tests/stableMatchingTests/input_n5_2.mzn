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

array [int, int] of 0..n-1: pm = [|3,1,2,0,4|4,2,1,0,3|1,4,0,3,2|4,1,3,2,0|3,0,1,2,4|];
array [int, int] of 0..n-1: pw = [|3,1,4,2,0|1,0,3,2,4|0,2,4,3,1|3,0,2,1,4|1,4,0,2,3|];

constraint stable_matching([m0, m1, m2, m3, m4], [w0, w1, w2, w3, w4], pm, pw) ::uniud;

solve satisfy;
