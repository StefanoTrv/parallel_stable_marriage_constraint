/*
 * mini-cp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * mini-cp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2018. by Laurent Michel, Pierre Schaus, Pascal Van Hentenryck
 */

#ifndef __HANDLE_H
#define __HANDLE_H

#include <memory>

template <typename T>
class handle_ptr {
   T* _ptr;
public:
   template <typename DT> friend class handle_ptr;
   typedef T element_type;
   typedef T* pointer;
   handle_ptr() noexcept : _ptr(nullptr) {}
   handle_ptr(T* ptr) noexcept : _ptr(ptr) {}
   handle_ptr(const handle_ptr<T>& ptr) noexcept : _ptr(ptr._ptr)  {}
   handle_ptr(std::nullptr_t ptr)  noexcept : _ptr(ptr) {}
   handle_ptr(handle_ptr<T>&& ptr) noexcept : _ptr(std::move(ptr._ptr)) {}
   template <typename DT> handle_ptr(DT* ptr) noexcept : _ptr(ptr) {}
   template <typename DT> handle_ptr(const handle_ptr<DT>& ptr) noexcept : _ptr(ptr.get()) {}
   template <typename DT> handle_ptr(handle_ptr<DT>&& ptr) noexcept : _ptr(std::move(ptr._ptr)) {}
   handle_ptr& operator=(const handle_ptr<T>& ptr) { _ptr = ptr._ptr;return *this;}
   handle_ptr& operator=(handle_ptr<T>&& ptr)      { _ptr = std::move(ptr._ptr);return *this;}
   handle_ptr& operator=(T* ptr)                   { _ptr = ptr;return *this;}
   T* get() const noexcept { return _ptr;}
   T* operator->() const noexcept { return _ptr;}
   T& operator*() const noexcept  { return *_ptr;}
   template<typename SET> SET& operator*() const noexcept {return *_ptr;}
   operator bool() const noexcept { return _ptr != nullptr;}
   void dealloc() { delete _ptr;_ptr = nullptr;}
   void free()    { delete _ptr;_ptr = nullptr;}
   template<class X> friend bool operator==(const handle_ptr<T>& p1,const handle_ptr<X>& p2)
   {
      return p1._ptr == p2.get();
   }
};

template <class T,class... Args>
inline handle_ptr<T> make_handle(Args&&... formals)
{
   return handle_ptr<T>(new T(std::forward<Args>(formals)...));
}

#endif
