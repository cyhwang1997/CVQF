#include "cuckoofilter.h"

#include <assert.h>
#include <math.h>

#include <iostream>
#include <vector>

#include <sys/time.h>
#include <time.h>
#include <chrono>
#include <openssl/rand.h>

uint64_t tv2usec(struct timeval *tv) {
   return 1000000 * tv->tv_sec + tv->tv_usec;
}

using cuckoofilter::CuckooFilter;

int main(int argc, char **argv) {
  //size_t total_items = 1000000;
  size_t total_items = 2097152;
  //size_t total_items = 131072;

  uint64_t load_factor = atoi(argv[1]);

  // Create a cuckoo filter where each item is of type size_t and
  // use 12 bits for each item:
  //    CuckooFilter<size_t, 12> filter(total_items);
  // To enable semi-sorting, define the storage of cuckoo filter to be
  // PackedTable, accepting keys of size_t type and making 13 bits
  // for each key:
  //   CuckooFilter<size_t, 13, cuckoofilter::PackedTable> filter(total_items);
  //CuckooFilter<size_t, 12> filter(total_items);
  CuckooFilter<uint64_t, 8> filter(total_items);

  struct timeval start, end;
  struct timezone tzp;
  uint64_t elapsed_usecs;
  double insertion_throughput;
  double positive_throughput;
  double negative_throughput;
  double remove_throughput;

  uint64_t *vals = (uint64_t*)malloc(total_items*sizeof(vals[0]));
  RAND_bytes((unsigned char *)vals, sizeof(*vals)*total_items);

  gettimeofday(&start, &tzp);
  // Insert items to this cuckoo filter
  size_t num_inserted = 0;
  for (size_t i = 0; i < load_factor*total_items / 100; i++, num_inserted++) {
    if (filter.Add(vals[i]) != cuckoofilter::Ok) {
      break;
    }
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  insertion_throughput = 1.0 * num_inserted / elapsed_usecs;
  
  printf("Insertion Throughput: %f Million operations / sec\n", insertion_throughput);


  gettimeofday(&start, &tzp);
  // Check if previously inserted items are in the filter, expected
  // true for all items
  for (size_t i = 0; i < num_inserted; i++) {
    assert(filter.Contain(vals[i]) == cuckoofilter::Ok);
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  positive_throughput = 1.0 * num_inserted / elapsed_usecs;

  printf("Positive Throughput: %f Million operations / sec\n", positive_throughput);

  // Check non-existing items, a few false positives expected
  gettimeofday(&start, &tzp);
  size_t total_queries = 0;
  size_t false_queries = 0;
  for (size_t i = 0; i < num_inserted; i++) {
  // for (size_t i = total_items; i < 2 * total_items; i++) {
    if (filter.Contain(i) == cuckoofilter::Ok) {
      false_queries++;
    }
    total_queries++;
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  negative_throughput = 1.0 * num_inserted / elapsed_usecs;

  printf("Negative Throughput: %f Million operations / sec\n", negative_throughput);

  // Output the measured false positive rate
  std::cout << "false positive rate is "
            << 100.0 * false_queries / total_queries << "%\n";

  gettimeofday(&start, &tzp);
  for (size_t i = 0; i < num_inserted; i++) {
     assert(filter.Delete(vals[i]) == cuckoofilter::Ok);
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  remove_throughput = 1.0 * num_inserted / elapsed_usecs;

  printf("Remove Throughput: %f Million operations / sec\n", remove_throughput);

  return 0;
}
