#ifndef __MINICPP_HASHTABLE_H
#define __MINICPP_HASHTABLE_H

#include <algorithm>
#include <iostream>
#include <functional>
#include <string>
#include "store.hpp"

template <class K,class T,class Hash = std::hash<K>,class Equal = std::equal_to<K>> class Hashtable {
   struct HTNode {
      K _key;
      T _data;
      HTNode* _next;
   };
   static constexpr const int _primes[] =
      {
       2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,
       163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,
       331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,
       523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,
       733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,
       947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,
       1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,
       1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,
       1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619
      };
   Pool::Ptr _pool;
   Hash _hash;
   Equal _equal;
   HTNode**  _tab;
   unsigned* _mgc;
   int  _mxs;
   unsigned _magic;
   unsigned _nbp;   // number of pairs
public:
   Hashtable(Pool::Ptr p,int sz) : _pool(p) {
      constexpr const int tsz = sizeof(_primes)/sizeof(int);
      int low=0,up = tsz - 1;
      while (low <= up) {
         int m = (low + up)/2;
         if (sz < _primes[m])
            up = m - 1;
         else if (sz > _primes[m])
            low = m + 1;
         else {
            low = up = m;
            break;
         }
      }
      _mxs = low >= tsz ? sz :  _primes[low];
      std::cout << "SIZE:" << _mxs << '\n';
      _tab = new (_pool) HTNode*[_mxs];
      _mgc = new (_pool) unsigned[_mxs];
       memset(_tab, '\0', sizeof(HTNode*)*_mxs);
       memset(_mgc, '\0', sizeof(unsigned)*_mxs);
      _magic = 0;
      _nbp = 0;
   }
   void insert(const K& key,const T& val) noexcept {
      int at = _hash(key) % _mxs;
      HTNode* head = _mgc[at]==_magic ? _tab[at] : nullptr;
      HTNode* cur = head;
      while (cur != nullptr) {
         if (_equal(cur->_key,key)) {
            cur->_data = val;
            return ;
         }
         cur = cur->_next;
      }
      _tab[at] = new (_pool) HTNode {key,val,head};
      _mgc[at] = _magic;
      ++_nbp;
   }
   bool get(const K& key,T& val) const noexcept {
      int at = _hash(key) % _mxs;
      HTNode* cur =  _mgc[at]==_magic ? _tab[at] : nullptr;
      while (cur != nullptr) {
         if (_equal(cur->_key,key)) {
            val = cur->_data;
            return true;
         }
         cur = cur->_next;
      }
      return false;      
   }
   unsigned size() const noexcept { return _nbp;}
   void clear() noexcept {
      ++_magic;
      _nbp = 0;
   }
};

#endif
