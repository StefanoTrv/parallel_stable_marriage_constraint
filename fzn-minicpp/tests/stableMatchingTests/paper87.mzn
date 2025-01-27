include "stable_matching.mzn";
include "minicpp.mzn";

int: n = 8;

var 0..n-1: m0;
var 0..n-1: m1;
var 0..n-1: m2;
var 0..n-1: m3;
var 0..n-1: m4;
var 0..n-1: m5;
var 0..n-1: m6;
var 0..n-1: m7;

var 0..n-1: w0;
var 0..n-1: w1;
var 0..n-1: w2;
var 0..n-1: w3;
var 0..n-1: w4;
var 0..n-1: w5;
var 0..n-1: w6;
var 0..n-1: w7;

array [int, int] of 0..n-1: pm = [|
    4,6,0,1,5,7,3,2|
    1,2,6,4,3,0,7,5|
    7,4,0,3,5,1,2,6|
    2,1,6,3,0,5,7,4|
    6,1,4,0,2,5,7,3|
    0,5,6,4,7,3,1,2|
    1,4,6,5,2,3,7,0|
    2,7,3,4,6,1,5,0|
|];
array [int, int] of 0..n-1: pw = [|
    4,2,6,5,0,1,7,3|
    7,5,2,4,6,1,0,3|
    0,4,5,1,3,7,6,2|
    7,6,2,1,3,0,4,5|
    5,3,6,2,7,0,1,4|
    1,7,4,3,5,2,6,0|
    6,4,1,0,7,5,3,2|
    6,3,0,4,1,2,5,7|
|];

constraint stable_matching(
    [m0, m1, m2, m3, m4, m5, m6, m7], 
    [w0, w1, w2, w3, w4, w5, w6, w7], 
    pm, pw
) ::uniud;

solve satisfy;
