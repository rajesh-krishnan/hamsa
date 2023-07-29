#ifndef _MMAP_HELPER_H
#define _MMAP_HELPER_H
#include <stddef.h>

void *mymap (size_t size);
void myunmap (void *ptr, size_t size);

#endif /* _MMAP_HELPER_H */
