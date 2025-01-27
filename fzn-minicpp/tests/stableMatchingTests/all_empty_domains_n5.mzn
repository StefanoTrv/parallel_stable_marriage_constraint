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
    0, 1, 2, 4, 3 | 
    0, 2, 3, 1, 4 | 
    4, 1, 3, 2, 0 | 
    3, 4, 1, 0, 2 | 
    2, 0, 1, 4, 3 | 
|];

array [int, int] of 0..n-1: pw = [|
    4, 2, 0, 1, 3 | 
    2, 3, 0, 4, 1 | 
    0, 3, 2, 4, 1 | 
    0, 1, 2, 4, 3 | 
    2, 3, 4, 0, 1 | 
|];


% Directly enforce domain constraints for men
constraint m0 in {0,1,2,3,4} diff {0,1,2,4};  % Man 0 cannot pair with women 0, 1, 2, or 4
constraint m1 in {0,1,2,3,4} diff {3,4};      % Man 1 cannot pair with women 3 or 4
constraint m2 in {0,1,2,3,4} diff {0,4};      % Man 2 cannot pair with women 0 or 4
constraint m3 in {0,1,2,3,4} diff {0,1,2,3,4}; % Man 3 cannot pair with anyone (no domain)
constraint m4 in {0,1,2,3,4} diff {0,1,2,3};  % Man 4 cannot pair with women 0, 1, 2, or 3

% Directly enforce domain constraints for women
constraint w0 in {0,1,2,3,4} diff {4};         % Woman 0 cannot pair with man 4
constraint w1 in {0,1,2,3,4} diff {0,1,2,3,4}; % Woman 1 cannot pair with anyone (no domain)
constraint w2 in {0,1,2,3,4} diff {2,3};       % Woman 2 cannot pair with men 2 or 3
constraint w3 in {0,1,2,3,4} diff {0};         % Woman 3 cannot pair with man 0
constraint w4 in {0,1,2,3,4} diff {1};         % Woman 4 cannot pair with man 1

constraint stable_matching([m0, m1, m2, m3, m4], [w0, w1, w2, w3, w4], pm, pw) ::uniud;

solve satisfy;
