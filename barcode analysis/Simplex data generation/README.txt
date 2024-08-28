Package contains 2 subfolders with different code parameters: classic or prime moves.

Example compilation line for every subfolder (in current directory with g++):
g++ .\ac_bfs.cpp

Run with parameter n:
on Windows:
.\a.exe n

on Linux:
.\a.out n

where n is maximal size of presentation. It will write n back on standard output.
Code is designed to work with relators size up to ~40 each.

Example execution line:
.\a.exe 14
.\a.out 14

Code will generate 4 files in current directory containing simplex description of generated graph and it's filtrations.