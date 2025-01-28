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
    2,0,3,1|
    3,0,1,2|
    1,0,3,2|
    2,0,1,3|
|];

array [int, int] of 0..n-1: pw = [|
    3,2,0,1|
    0,3,1,2|
    2,3,1,0|
    3,2,1,0|
|];

constraint w0 != 1;
constraint w1 != 3;

constraint stable_matching([m0, m1, m2, m3], [w0, w1, w2, w3], pm, pw) ::uniud;

solve satisfy;
