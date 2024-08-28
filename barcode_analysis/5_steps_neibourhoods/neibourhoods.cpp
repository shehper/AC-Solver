#include "AC_UTILS_no_hash.cpp"
#include <iostream>
#include <fstream>
#include <queue>
#include <map>
#include <chrono>
#include <string>
#include <sstream>
#include <climits>

using namespace std;

// Function calculates size of 5 step neibourhood via BFS algorithm
// Parameters:
//  - 'start' : starting presentation
//  - 'radius' : how many steps
//  - 'classic' : bool, true if we are using classic moves, false if prime moves
int neibourhood (Presentation start, size_t radius, bool classic) {

    int mc_ = 12;
    if(classic) {
        mc_ = 14;
    }
    
    queue<Presentation> que;
    set<Presentation> visited;
    que.push(start);
    visited.insert(start);
    unsigned long long size_ = 0;
    size_ ++;
    map<Presentation, unsigned long long> dist;
    dist[start] = 0;

    while(!que.empty()) {
        auto current = que.front();
        que.pop();

        for (int k = 0; k < mc_; k ++) {
            auto child = move(current, k, classic);
            unsigned long long r = dist[current];
        
            if ((r < radius) && (visited.find(child) == visited.end())) {
                que.push(child);
                visited.insert(child);
                dist[child] = r + 1;
                size_ ++;
                
                assert(size_ < ULLONG_MAX);
            }
       }

    }
    return size_;
}

// Reads presentations from 'sourcefile', calculates sizes of neibourhoods and writes it to 'outfile'.
// Neibourhoods parameters: radius 'r' and bool 'classic'.
int read_do_and_write(string sourcefile, string outfile, size_t r, bool classic){
    std::ifstream inputFile(sourcefile);
    std::ofstream outputFile(outfile);
    std::string line;

    if (inputFile.is_open()) {
        while (std::getline(inputFile, line)) {
                std::vector<int> data;
                std::stringstream ss(line);
                int number;
                char comma;  // To skip commas if present in the data
                ss >> comma;
                while (ss >> number) {
                    data.push_back(number);
                    ss >> comma;  // This will consume the comma, allowing the next integer to be read
                }
            

                Relator rel1;
                Relator rel2;

                for(int i = 0; i < data.size()/2;i ++) {
                    if(data[i] != 0) {
                        rel1.push_back(data[i]);
                    }
                }
                for(int i = data.size()/2; i < data.size();i ++) {
                    if(data[i] != 0) {
                        rel2.push_back(data[i]);
                    }
                }
                cout << sort_(rel1, rel2) << endl;
                outputFile << neibourhood(sort_(rel1, rel2), r, classic) << "\n" << flush;
            }
            
        inputFile.close();
        outputFile.close();
    } else {
        std::cerr << "Unable to open file";
        return 1;
    }

    return 0;
}

int main() {
    size_t radi = 5;

    string solved = "solved_miller_schupp_presentations.txt";
    string unsolved = "unsolved_miller_schupp_presentations.txt";

    string solved_out_p = "solved_prime_moves";
    string unsolved_out_p = "unsolved_prime_moves";
    string unsolved_out_c = "unsolved_clasic_moves";
    string solved_out_c = "solved_clasic_moves";

    
    read_do_and_write(solved, solved_out_c, radi, true);
    read_do_and_write(unsolved, unsolved_out_c, radi, true);

    read_do_and_write(solved, solved_out_p, radi, false);
    read_do_and_write(unsolved, unsolved_out_p, radi, false);


    // read_do_and_write("test_input.txt","test_output.txt", radi, false); 
}

