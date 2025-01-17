#ifndef __VARITF_H
#define __VARITF_H

#include "avar.hpp"
#include "store.hpp"
#include "solver.hpp"
#include "trailList.hpp"

template<> class var<int> : public AVar {
   friend class Storage;
private:
   int _id;
protected:
   void setId(int id) override { _id = id;}
public:
   int getId() const noexcept { return _id;}
   typedef handle_ptr<var<int>> Ptr;
   virtual Storage::Ptr getStore() = 0;
   virtual CPSolver::Ptr getSolver() = 0;
   virtual int min() const  = 0;
   virtual int max() const  = 0;
   virtual int size() const = 0;
   virtual bool isBound() const = 0;
   virtual bool changed() const noexcept = 0;
   virtual bool changedMax() const noexcept = 0;
   virtual bool changedMin() const noexcept = 0;
   virtual bool contains(int v) const = 0;
   virtual bool containsBase(int v) const { return contains(v);}
   virtual void dump(int min, int max, unsigned int * dump) const {throw std::runtime_error("Unsupported opration");};
   virtual int getIthVal(int index) const { throw std::runtime_error("Unsupported opration");};
   virtual void assign(int v) = 0;
   virtual void remove(int v) = 0;
   virtual void removeBelow(int newMin) = 0;
   virtual void removeAbove(int newMax) = 0;
   virtual void updateBounds(int newMin,int newMax) = 0;

   virtual TLCNode* whenBind(std::function<void(void)>&& f) = 0;
   virtual TLCNode* whenBoundsChange(std::function<void(void)>&& f) = 0;
   virtual TLCNode* whenDomainChange(std::function<void(void)>&& f) = 0;
   virtual TLCNode* propagateOnBind(Constraint::Ptr c)          = 0;
   virtual TLCNode* propagateOnBoundChange(Constraint::Ptr c)   = 0;
   virtual TLCNode* propagateOnDomainChange(Constraint::Ptr c ) = 0;
   virtual std::ostream& print(std::ostream& os) const = 0;
   friend std::ostream& operator<<(std::ostream& os,const var<int>& x) { return x.print(os);}
};


#endif
