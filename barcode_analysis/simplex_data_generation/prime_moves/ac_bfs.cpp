#include "AC_UTILS_as_sets.h"
#include <iostream>
#include <fstream>
#include <queue>
#include <map>
#include <chrono>
#include <string>
#include <climits>

using namespace std;

int main (int argc, char *argv[]) {

    int n = atoi(argv[1]);
    cout << n <<endl;

    // Duration counter start
    auto start_t = chrono::high_resolution_clock::now();

    string file_name_1F = "one_filtrations_" + std::to_string(n);
    ofstream one_f(file_name_1F);
    one_f << "{\"1-filt\":[";

    string file_name_0F = "zero_filtrations_" + std::to_string(n);
    ofstream zero_f(file_name_0F);
    zero_f << "{\"0-filt\":[";

    string file_name_1S = "one_simplices_" + std::to_string(n);
    ofstream one_s(file_name_1S);
    one_s << "{\"1-simplices\":[";

    string file_name_0S = "zero_simplices_" + std::to_string(n);
    ofstream zero_s(file_name_0S);
    zero_s << "{\"0-simplices\":[";

    auto start = Presentation({1}, {2});

    

    // BFS
    queue<Presentation> que;
    set<Presentation> visited;
    que.push(start);
    visited.insert(start);
    map<Presentation, unsigned long long> name_;
    unsigned long long inx = 0;
    name_[start] = 0;
    zero_s << '[' << inx << ']' << ',';                
    zero_f << start.size()  << ',';  
    inx ++;

    while(!que.empty()) {
        auto current = que.front();
        que.pop();

        for (int k = 0; k < 12; k ++) {
            auto child = current.move(k);

            if (child.size() <= n && visited.find(child)== visited.end()) {
                que.push(child);
                visited.insert(child);
                name_[child] = inx;

                zero_s << '[' << inx << ']' << ',';                
                zero_f << child.size()  << ',';                

                inx ++;
                assert(inx < ULLONG_MAX);
            }

            if(child.size() <= n) {
                auto cn = name_[current];
                auto cc = name_[child];
                if(cn < cc) {
                    one_f << std::max(current.size(), child.size()) << ',';
                    one_s << "[" << cn << "," << cc << "],";
                }
            }

       }
    }

	one_f << "-5]}";
	one_s << "[]]}";
    zero_f << "-5]}";
    zero_s << "[]]}";
    
    one_f.close();
    one_f.close();
    zero_s.close();
    zero_f.close();

    // Duration counter stop
	auto stop_t = chrono::high_resolution_clock::now();

	auto duration = chrono::duration_cast<chrono::microseconds>(stop_t-start_t);
	std::cout << "time " << duration.count() << endl;

}