#ifndef AC_H
#define AC_H

#include <set>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <string>

#define a 1
#define A -1
#define b 2
#define B -2

typedef std::vector<int> Relator;

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

class Hash;
class Presentation;

//
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

int next(int gen, int type) {
    assert(gen == A || gen == b || gen == a || gen == B);
    assert(type == 0 || type == 1 || type == 2);

    if(gen == a){
        if(type == 0) return a;
        else if (type == 1) return b;
        else return B;
    }
    else if(gen == A){
        if(type == 0) return A;
        else if (type == 1) return b;
        else return B;
    }
    else if(gen == b){
        if(type == 0) return a;
        else if (type == 1) return A;
        else return b;
    }
    else if(gen == B){
        if(type == 0) return a;
        else if (type == 1) return A;
        else return B;
    }
    else return 0;
}

bool help1_(int gen1, int gen2, Relator x, Relator y) {
    for(unsigned int i = 0; i < x.size(); i ++){
        if(gen1 == x[i] && gen2 == y[i])
            return true;
    }
    return false;
}

unsigned long long type(int gen1, int gen2) {
    assert(gen1 == A || gen1 == b || gen1 == a || gen1 == B);
    assert(gen2 == A || gen2 == b || gen2 == a || gen2 == B);

    if(help1_(gen1, gen2, {a, A, b, B}, {a, A, a, a}))
        return 0;
    if(help1_(gen1, gen2, {a, A, b, B}, {b, b, A, A}))
        return 1;
    if(help1_(gen1, gen2, {a, A, b, B}, {B, B, b, B}))
        return 2;
    assert(false);
}



// Presentation and its hash:
// both are printable and with reasonable contructors
// Hash from relators and into pair of relators
// Presentation is being kept as its hash

class Hash {
  public:
    Hash() : len1_(1), len2_(1), start1_(a), start2_(b), number1_(0), number2_(0) {}
    Hash(int len1, int len2, int start1, int start2, unsigned long long number1, unsigned long long number2) : len1_(len1), len2_(len2), start1_(start1), start2_(start2), number1_(number1), number2_(number2) {
        assert(len1 <= 40 && len2 <= 40);
    }
    Hash(const Hash& hash) = default;
    Hash(Relator rel1, Relator rel2) {
        assert(rel1.size() <= 40 && rel2.size() <= 40);
        unsigned long long wyk = 1;
        unsigned long long number1 = 0;
        unsigned long long number2 = 0;
        for(unsigned long long i = 0; i < rel1.size() - 1; i ++) {
            number1 += wyk * type(rel1[i], rel1[i + 1]);
            wyk *= 3;
        }
        wyk = 1;
        for(unsigned long long i = 0; i < rel2.size() - 1; i ++) {
            number2 += wyk * type(rel2[i], rel2[i + 1]);
            wyk *= 3;
        }
        len1_ = rel1.size();
        len2_ = rel2.size();
        start1_ = rel1[0];
        start2_ = rel2[0];
        number1_ = number1;
        number2_ = number2;
    }

    ~Hash() = default;

    Hash& operator=(const Hash& other) = default;
    Hash& operator=(Hash&& other) = default;
   

    std::pair<Relator, Relator> ToRelators() {
        assert(len1_ > 0 && len2_ > 0);
        Relator rel1(len1_, 0);
        Relator rel2(len2_, 0);
        rel1[0] = start1_;
        rel2[0] = start2_;
        unsigned long long number1 = number1_;
        unsigned long long number2 = number2_;
        for(int i = 1; i < len1_; i++) {
            rel1[i] = next(rel1[i-1], number1  % 3);
            number1 /= 3;
        }
        for(int i = 1; i < len2_; i++) {
            rel2[i] = next(rel2[i-1], number2 % 3);
            number2 /= 3;
        }

        assert(number1 == 0);
        assert(number2 == 0);

        return std::make_pair(rel1, rel2);
    }

    size_t size() {
        return len1_ + len2_;
    }

    int len1_;
    int len2_;
    int start1_;
    int start2_;
    unsigned long long number1_;
    unsigned long long number2_;
};

bool operator<(const Hash& Left, const Hash& Right){
        if(Left.len1_ < Right.len1_) return true;
        else if (Left.len1_ > Right.len1_) return false;
        else if(Left.len2_ < Right.len2_) return true;
        else if(Left.len2_ > Right.len2_) return false;
        else if(Left.start1_ < Right.start1_) return true;
        else if(Left.start1_ > Right.start1_) return false;
        else if(Left.start2_ < Right.start2_) return true;
        else if(Left.start2_ > Right.start2_) return false;
        else if(Left.number1_ < Right.number1_) return true;
        else if(Left.number1_ > Right.number1_) return false;
        else return Left.number2_ < Right.number2_;
}

bool operator==(const Hash& rhs, const Hash& other) {
    return rhs.len1_ == other.len1_ && rhs.len2_ == other.len2_ && rhs.start1_ == other.start1_ && rhs.start2_ == other.start2_ && rhs.number1_ == other.number1_ && rhs.number2_ == other.number2_;
}
bool operator!=(const Hash& lhs, const Hash& rhs) {
    return !(rhs == lhs);
}

 std::ostream & operator<<(std::ostream& os, Hash hash)
    {
      os << '(' << hash.len1_ << ", " << hash.len2_ << ", " << hash.start1_ << ", " << hash.start2_ << ", " << hash.number1_ << ", " << hash.number2_ << ')';
      return os;
    }

std::ostream & operator<<(std::ostream& os, Relator rel) {
    for(unsigned int i = 0; i < rel.size(); i ++) {
        os << toChar(rel[i]);
    }
    return os;
}

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

// Presentation with implementation of moves:
class Presentation {
    public:
        Hash hash_;
        Presentation(Hash hash) : hash_(hash) {}
        Presentation() = default;
        Presentation(Relator rel1, Relator rel2) {
            if(rel1 < rel2) {
                hash_ = Hash(rel1, rel2);
            } else {
                hash_ = Hash(rel2, rel1);
            }
        }
        Presentation(std::pair<Relator, Relator> rels) {
            if(rels.first < rels.second) {
                hash_ = Hash(rels.first, rels.second);
            } else {
                hash_ = Hash(rels.second, rels.first);
            }
        }
        ~Presentation() = default;
        Presentation(const Presentation& pres) = default;

        size_t size() {
            return hash_.size();
        }

        Presentation move(int t) {
            auto rels = hash_.ToRelators();
            Relator rel1 = rels.first;
            Relator rel2 = rels.second;

            // if(t == 0)
            //     return Presentation(concat_(rel1, rel2), rel2);
            // else if(t == 1)
            //     return Presentation(concat_(rel2, rel1), rel2);
            // else if(t == 2) 
            //     return Presentation(rel1, concat_(rel1, rel2));
            // else if(t == 3)
            //     return Presentation(rel1, concat_(rel2, rel1));
            // else if(t == 4)
            //     return Presentation(rel1, conj0_(rel2, a));
            // else if(t == 5)
            //     return Presentation(rel1, conj0_(rel2, b));
            // else if(t == 6)
            //     return Presentation(rel1, conj0_(rel2, A));
            // else if(t == 7)
            //     return Presentation(rel1, conj0_(rel2, B));
            // else if(t == 8)
            //     return Presentation(conj0_(rel1, a), rel2);
            // else if(t == 9)
            //     return Presentation(conj0_(rel1, b), rel2);
            // else if(t == 10)
            //     return Presentation(conj0_(rel1, A), rel2);
            // else if(t == 11)
            //     return Presentation(conj0_(rel1, B), rel2);
            // else if(t == 12)
            //     return Presentation(inv0_(rel1), rel2);
            // else if(t == 13)
            //     return Presentation(rel1, inv0_(rel2));
            // else if(t == 14)
            //     return Presentation(rel2, rel1);
            // else
            //     assert(false);

            // Prime moves:
            if(t == 0)
                return Presentation(concat_(rel1, rel2), rel2);
            else if(t == 1)
                return Presentation(rel1, concat_(rel2, rel1));
            else if(t == 2)
                return Presentation(concat_(rel1, inv0_(rel2)), rel2);
            else if(t == 3)
                return Presentation(rel1, concat_(rel2, inv0_(rel1)));
            else if(t == 4)
                return Presentation(conj0_(rel1, B), rel2);
            else if(t == 5)
                return Presentation(conj0_(rel1, A), rel2);
            else if(t == 6)
                return Presentation(conj0_(rel1, a), rel2);
            else if(t == 7)
                return Presentation(conj0_(rel1, b), rel2);
            else if(t == 8)
                return Presentation(rel1, conj0_(rel2, B));
            else if(t == 9)
                return Presentation(rel1, conj0_(rel2, A));
            else if(t == 10)
                return Presentation(rel1, conj0_(rel2, a));
            else if(t == 11)
                return Presentation(rel1, conj0_(rel2, b));
            else
                assert(false);
        }

    std::pair<Relator, Relator> relators() {
        return hash_.ToRelators();
    }

    // does not make moves that go out of range // wont work with jessicas code
    std::set<Presentation> neibours(){
        std::set<Presentation> s;
        if(hash_.len1_ + hash_.len2_ <= 40) {
            for(int i = 0; i < 4; i ++)
                s.insert(move(i));
        }

        if(hash_.len2_ <= 38) {
            for(int i = 4; i < 8; i ++)
                s.insert(move(i));
        }
        if(hash_.len1_ <= 38) {
            for(int i = 8; i < 12; i ++)
                s.insert(move(i));
        }

        for(int i = 12; i < 15; i ++)
                s.insert(move(i));

        return s;
    }

    std::vector<Presentation> neibours_v(){
        auto s = neibours();
        std::vector<Presentation> vec;
        for(auto p : s)
            vec.push_back(p);
        return vec;
    }

};

bool operator<(const Presentation& Left, const Presentation& Right){
    return Left.hash_ < Right.hash_;
}

bool operator==(const Presentation& rhs, const Presentation& other) {
    return rhs.hash_ == other.hash_;
}

bool operator!=(const Presentation& rhs, const Presentation& other) {
    return rhs.hash_ != other.hash_;
}


// std::ostream & operator<<(std::ostream& os, Presentation& pres) {
//     auto rels = pres.hash_.ToRelators();
//     os << '(' << rels.first << ',' << ' ' << rels.second << ')';
//     return os;
// }

std::ostream & operator<<(std::ostream& os, Presentation pres) {
    auto rels = pres.hash_.ToRelators();
    os << '(' << rels.first << ',' << ' ' << rels.second << ')';
    return os;
}

#endif //AC_H
