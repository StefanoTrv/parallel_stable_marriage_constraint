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

#include "mallocWatch.hpp"

#if defined(__x86_64__) && defined(__APPLE__)

#include <mach/mach.h> 
#include <mach/vm_map.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <malloc/malloc.h>
#include <stdarg.h>

extern void _simple_vdprintf(int, const char *, va_list);
inline void nomalloc_printf(const char *format, ...)
{
   va_list ap;
   va_start(ap, format);
   _simple_vdprintf(STDOUT_FILENO, format, ap);
   va_end(ap);
}

void *(*system_malloc)(malloc_zone_t *zone, size_t size);
void *(*system_valloc)(malloc_zone_t *zone, size_t size);
void *(*system_calloc)(malloc_zone_t *zone, size_t size,size_t cnt);
void *(*system_realloc)(malloc_zone_t* zone,void* ptr,size_t size);
void (*system_free)(malloc_zone_t *zone, void *ptr);
void (*system_free_definite_size)(malloc_zone_t * zone, void *ptr , size_t sz);

static long nbBytes = 0;
static long peakBytes = 0;

void * my_malloc(malloc_zone_t *zone, size_t size)
{
   void *ptr = system_malloc(zone, size);
   size_t als = malloc_size(ptr);
   nbBytes += als;
   peakBytes = nbBytes > peakBytes ? nbBytes : peakBytes;
   return ptr;
}

void * my_valloc(malloc_zone_t *zone, size_t size)
{
   void *ptr = system_valloc(zone, size);
   size_t als = malloc_size(ptr);
   nbBytes += als;
   peakBytes = nbBytes > peakBytes ? nbBytes : peakBytes;
   return ptr;
}

void *my_realloc(malloc_zone_t* zone,void* ptr,size_t size)
{
   size_t oldsz = malloc_size(ptr);
   void* nPtr = system_realloc(zone,ptr,size);
   size_t newsz = malloc_size(nPtr);
   nbBytes += newsz - oldsz;
   return nPtr;
}

void * my_calloc(malloc_zone_t *zone, size_t cnt,size_t size)
{
   void* ptr = system_calloc(zone,cnt,size);
   size_t als = malloc_size(ptr);
   nbBytes += als;
   peakBytes = nbBytes > peakBytes ? nbBytes : peakBytes;
   return ptr;
}

void my_free(malloc_zone_t *zone, void *ptr)
{
   size_t toFree = malloc_size(ptr);
   nbBytes -= toFree;
   system_free(zone, ptr);
}

void my_free_definite_size(malloc_zone_t * zone, void *ptr , size_t sz)
{
   nbBytes -= sz;
   system_free_definite_size(zone, ptr,sz);
}

void mallocWatch()
{
   size_t  protect_size = sizeof(malloc_zone_t);
   malloc_zone_t *zone = malloc_default_zone();
   if (zone->malloc == my_malloc) return ;
   system_malloc = zone->malloc;
   system_valloc = zone->valloc;
   system_calloc = zone->calloc;
   system_realloc = zone->realloc;
   system_free = zone->free;  // ignoring atomicity/caching
   system_free_definite_size = zone->free_definite_size;
   
   if(zone->version >= 8) {
      vm_protect(mach_task_self(), (uintptr_t)zone, protect_size, 0, VM_PROT_READ | VM_PROT_WRITE);//remove the write protection
   }
   zone->malloc = my_malloc;
   zone->valloc = my_valloc;
   zone->calloc = my_calloc;
   zone->realloc = my_realloc;
   zone->free = my_free;
   zone->free_definite_size = my_free_definite_size;
   if(zone->version==8) {
      vm_protect(mach_task_self(), (uintptr_t)zone, protect_size, 0, VM_PROT_READ);//put the write protection back
   }
}

void mallocReport()
{
   printf("malloc usage  %ld, %ld\n",nbBytes,peakBytes); 
}

#else

#include <stdlib.h>
#include <stdio.h>

void mallocWatch()
{}
void mallocReport()
{
  printf("iOS no report\n");
}

#endif
