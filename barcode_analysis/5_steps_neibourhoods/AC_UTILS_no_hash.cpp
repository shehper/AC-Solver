#ifndef AC_H
#define AC_H

#include <set>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <string>
#include<algorithm>

#define a 1
#define A -1
#define b 2
#define B -2

typedef std::vector<int> Relator;
typedef std::pair<Relator, Relator> Presentation;

bool operator<(const Relator& Left, const Relator& Right){
    if(Left.size() != Right.size()) {
        return (Left.size() < Right.size());
    }
    else {
        size_t i = 0;
        size_t n = Left.size();
        while(i < n && Left[i] == Right[i]) {
            i ++;
        }

        if(i == n) {
            return 0;
        }
        else {
            return Left[i] < Right[i];
        }
    }
}


char toChar(int g){
    if(g == a)
        return 'a';
    else if(g == A)
        return 'A';
    else if(g == b)
        return 'b';
    else if(g == B)
        return 'B';

    assert(false);
}

std::ostream & operator<<(std::ostream& os, Relator rel) {
    for(unsigned int i = 0; i < rel.size(); i ++) {
        os << toChar(rel[i]);
    }
    return os;
}

//  Free reduction of relators:
Relator reduce0_(Relator rel) {
    if(rel.size() < 2) return rel;
    Relator out;
    unsigned int j = 0;
    while(j < rel.size() - 1){
        if(rel[j] + rel[j + 1] == 0){
            j +=2;
        }
        else {
            out.push_back(rel[j]);
            j ++;
        }
    }

    if( rel[rel.size() - 1] + rel[rel.size() - 2] != 0){
        out.push_back(rel[rel.size() - 1]);
    }

    return out;
}

Relator reduce_(Relator rel) {
    Relator out = reduce0_(rel);
    while(out != rel) {
        rel = out;
        out = reduce0_(rel);
    }
    return out;
}

//  Moves on relators
Relator conj0_(Relator rel, int x) {
    Relator l(rel.size() + 2, 0);
    l[0] = (-1) * x;
    for(unsigned int i = 0; i < rel.size(); i++) {
        l[i + 1] = rel[i];
    }
    l[l.size() - 1] = x;
    return reduce_(l);
}

Relator inv0_(Relator rel) {
    Relator out = Relator(rel.size(), 0);
    for(unsigned int i = 0; i < rel.size(); i++) {
        out[i] = (-1) * rel[rel.size() - i - 1];
    }
    return out;
}

Relator concat_(Relator rel1, Relator rel2) {
    Relator rel(rel1.size() + rel2.size(), 0);
    for(unsigned int i = 0; i < rel1.size(); i ++)
        rel[i] = rel1[i];
    for(unsigned int i = rel1.size(); i < rel.size(); i ++)
        rel[i] = rel2[i - rel1.size()];
    return reduce_(rel);
}

Presentation sort_(Presentation p) {
    if(p.first < p.second) {
        return p;
    }
    else {
        return std::make_pair(p.second, p.first);
    }
}

Presentation sort_(Relator r1, Relator r2) {
    if(r1 < r2) {
        return std::make_pair(r1, r2);
    }
    else {
        return std::make_pair(r2, r1);
    }
}

size_t size(Presentation pres) {
    return pres.first.size() + pres.second.size();
}

// AC-moves on presentation 'pres', move's number 't' and if we are using classir or prime moves 'classic'

Presentation move(Presentation pres, int t, bool classic) {
            Presentation sp = sort_(pres);
            Relator rel1(sp.first.begin(), sp.first.end());
            Relator rel2(sp.second.begin(), sp.second.end());
            if(classic){
                // classic moves (14)
                if(t == 0)
                    return sort_(concat_(rel1, rel2), rel2);
                else if(t == 1)
                    return sort_(concat_(rel2, rel1), rel2);
                else if(t == 2) 
                    return sort_(rel1, concat_(rel1, rel2));
                else if(t == 3)
                    return sort_(rel1, concat_(rel2, rel1));
                else if(t == 4)
                    return sort_(rel1, conj0_(rel2, a));
                else if(t == 5)
                    return sort_(rel1, conj0_(rel2, b));
                else if(t == 6)
                    return sort_(rel1, conj0_(rel2, A));
                else if(t == 7)
                    return sort_(rel1, conj0_(rel2, B));
                else if(t == 8)
                    return sort_(conj0_(rel1, a), rel2);
                else if(t == 9)
                    return sort_(conj0_(rel1, b), rel2);
                else if(t == 10)
                    return sort_(conj0_(rel1, A), rel2);
                else if(t == 11)
                    return sort_(conj0_(rel1, B), rel2);
                else if(t == 12)
                    return sort_(inv0_(rel1), rel2);
                else if(t == 13)
                    return sort_(rel1, inv0_(rel2));
                else if(t == 14)
                    return sort_(rel2, rel1);
                else
                    assert(false);
            } else {
                 // Prime moves (12):
                if(t == 0)
                    return sort_(concat_(rel1, rel2), rel2);
                else if(t == 1)
                    return sort_(rel1, concat_(rel2, rel1));
                else if(t == 2)
                    return sort_(concat_(rel1, inv0_(rel2)), rel2);
                else if(t == 3)
                    return sort_(rel1, concat_(rel2, inv0_(rel1)));
                else if(t == 4)
                    return sort_(conj0_(rel1, B), rel2);
                else if(t == 5)
                    return sort_(conj0_(rel1, A), rel2);
                else if(t == 6)
                    return sort_(conj0_(rel1, a), rel2);
                else if(t == 7)
                    return sort_(conj0_(rel1, b), rel2);
                else if(t == 8)
                    return sort_(rel1, conj0_(rel2, B));
                else if(t == 9)
                    return sort_(rel1, conj0_(rel2, A));
                else if(t == 10)
                    return sort_(rel1, conj0_(rel2, a));
                else if(t == 11)
                    return sort_(rel1, conj0_(rel2, b));
                else
                    assert(false);
                }
        }

std::ostream & operator<<(std::ostream& os, Presentation pres) {
    os << '(' << pres.first << ',' << ' ' << pres.second << ')';
    return os;
}

#endif //AC_H
