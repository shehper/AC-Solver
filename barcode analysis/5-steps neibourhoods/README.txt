In the current directory there are supposed to be four files:
- 2 program files: "neibourhoods.cpp" and "AC_UTILS_no_hash.cpp"
- 2 input data files: "solved_miller_schupp_presentations.txt" and "unsolved_miller_schupp_presentations.txt"

Example compilation line (in the current directory with g++):
g++ .\neibourhoods.cpp

Run:
on Windows:
.\a.exe

on Linux:
.\a.out

During the run program will print out on standard output a presentation, that is currently being computed.


For testing:
you can comment lines 109, 110, 112, 113 in neibourhoods.cpp
and use line 116.
To do this you need to include the text file "test_input.txt" in the current directory
with one proper presentation (list of proper numbers of even length) per line.

Example:
"test_input.txt":
[1, 0, 0, 0, 2, 0, 0, 0]
[1, 2, 1, 0, 2, 0, 0, 0]
[1, 2, 1,-2, 0, 0, 2, 1, -2, 0, 0, 0]
[1, 2]
[2, 1]

Then "test_output.txt" will contain a list of numbers:
28631
49668
72392
28631
28631

And will print on standard output:
(a, b)
(b, aba)
(baB, abaB)
(a, b)
(a, b)
