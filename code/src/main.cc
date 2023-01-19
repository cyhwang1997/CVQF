/*
 * ============================================================================
 *
 *       Filename:  main.cc
 *
 *         Author:  Prashant Pandey (), ppandey2@cs.cmu.edu
 *   Organization:  Carnegie Mellon University
 *
 * ============================================================================
 */

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>
#include <openssl/rand.h>
#include <sys/time.h>

#include <set>
#include <time.h>

#include "vqf_filter.h"

// FOR ZIPFIAN
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include "zipfian_int_distribution.h"

#define TEST_NUM 1

// MD5
#include <openssl/md5.h>

#ifdef __AVX512BW__
extern __m512i SHUFFLE [];
extern __m512i SHUFFLE_REMOVE [];
extern __m512i SHUFFLE16 [];
extern __m512i SHUFFLE_REMOVE16 [];
#endif

uint64_t tv2usec(struct timeval *tv) {
  return 1000000 * tv->tv_sec + tv->tv_usec;
}

/* Print elapsed time using the start and end timeval */
void print_time_elapsed(const char* desc, struct timeval* start, struct
      timeval* end, uint64_t ops, const char *opname) {
  uint64_t elapsed_usecs = tv2usec(end) - tv2usec(start);
  printf("%s Total Time Elapsed: %f seconds", desc, 1.0*elapsed_usecs / 1000000);
  if (ops) {
    printf(" (%f nanoseconds/%s)", 1000.0 * elapsed_usecs / ops, opname);
    printf(" [%f Million operations / sec]", 1.0 * ops / elapsed_usecs);
  }
  printf("\n");
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "Please specify three arguments: \n \
                     1. Log2 of the number of slots in the CQF.\n \
                     2. Load factor (0 - 95).\n \
                     3. Skewness (0 - 99).\n \
                     4. ");
    exit(1);
  }

  uint64_t qbits = atoi(argv[1]);
  uint64_t nslots = (1ULL << qbits);
  uint64_t load_factor = atoi(argv[2]);
  uint64_t nvals = load_factor*nslots/100;
  uint64_t *vals;
  uint64_t *other_vals;
  uint64_t skewness = atoi(argv[3]);

  uint64_t sizeval = atoi(argv[4]);
  double sizevalR = 1.0 * sizeval / 100;

  double insertion_throughput = 0.0;
  double positive_throughput = 0.0;
  double negative_throughput = 0.0;
  double remove_throughput = 0.0;

/*
#if TAG_BITS == 8
   uint64_t filterMetadataRange = (nslots + 48)/48;
   filterMetadataRange = filterMetadataRange * 80 * (1ULL << TAG_BITS);
#elif TAG_BITS == 16
   uint64_t filterMetadataRange = (nslots + 28)/28;
   filterMetadataRange = filterMetadataRange * 36 * (1ULL << TAG_BITS);
#endif
   printf("filter->metadata.range = %ld, nvals = %ld, nslots = %ld\n", filterMetadataRange, nvals, nslots);
*/

  /* Repeat the test for TEST_NUM times. */
  for (int test_num = 0; test_num < TEST_NUM; test_num++) {
    vqf_filter *filter;	

    /* initialize vqf filter */
    if ((filter = vqf_init(sizevalR*nslots)) == NULL) {
      fprintf(stderr, "Can't allocate vqf filter.");
      exit(EXIT_FAILURE);
    }

    if (skewness == 0) {
      // Generate random values
      vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
      RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
      for (uint64_t i = 0; i < nvals; i++) {
        vals[i] = (1 * vals[i]) % filter->metadata.range;
      }
    } else if (skewness > 99) {
      fprintf(stderr, "Such skewness not allowed\n");
      exit(EXIT_FAILURE);
    } else {
      uint64_t *foo;
      foo = (uint64_t*)malloc((filter->metadata.range)*sizeof(foo[0]));
      for (uint64_t i = 0; i < filter->metadata.range; i++) {
        foo[i] = i;
      }
      unsigned seed;
      seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle(foo, foo + filter->metadata.range, std::default_random_engine(seed));

      printf("SHUFFLED\n");

      float skewnessR = skewness / 100.0;
      uint64_t pre_val;
      std::default_random_engine generator;
      vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
      zipfian_int_distribution<uint64_t> distribution(0, nvals, skewnessR);
      for (uint64_t i = 0; i < nvals; i++) {
        pre_val = distribution(generator) % filter->metadata.range;
        vals[i] = foo[pre_val];
      }

      printf("Zipfian Created\n");
    }

    other_vals = (uint64_t*)malloc(nvals*sizeof(other_vals[0]));
    RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nvals);
    for (uint64_t i = 0; i < nvals; i++) {
      other_vals[i] = (1 * other_vals[i]) % filter->metadata.range;
    }

    struct timeval start, end;
    struct timezone tzp;
    uint64_t elapsed_usecs;

    int insert_return = 1;

    // nanos
    //struct timespec startT, endT;

    //int put_slot = 0;
    //int all_slot = 0;

    gettimeofday(&start, &tzp);
    /* Insert hashes in the vqf filter */
    for (uint64_t i = 0; i < nvals; i++) {
      //clock_gettime(CLOCK_MONOTONIC, &startT);
      //vqf_insert(filter, vals[i]);
      //clock_gettime(CLOCK_MONOTONIC, &endT);
      //printf("%ld\n", 1000000000 * (endT.tv_sec - startT.tv_sec) + (endT.tv_nsec - startT.tv_nsec));
      insert_return = vqf_insert(filter, vals[i]);
      //put_slot += insert_return;
      //all_slot++;
      if (insert_return == -1) {
        fprintf(stderr, "Insertion failed\n");
        exit(EXIT_FAILURE);
      }
    }
    gettimeofday(&end, &tzp);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    insertion_throughput += 1.0 * nvals / elapsed_usecs;
    print_time_elapsed("Insertion time", &start, &end, nvals, "insert");
    //printf("\n%d/%d\n\n", put_slot, all_slot);

    gettimeofday(&start, &tzp);
    /* Lookup hashes in the vqf filter (Successful Lookup) */
    for (uint64_t i = 0; i < nvals; i++) {
      if (!vqf_is_present(filter, vals[i])) {
        fprintf(stderr, "Lookup failed for %ld\n", vals[i]);
        exit(EXIT_FAILURE);
      }
    }
    gettimeofday(&end, &tzp);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    positive_throughput += 1.0 * nvals / elapsed_usecs;
      
    print_time_elapsed("Lookup time", &start, &end, nvals, "successful lookup");
      
      
    gettimeofday(&start, &tzp);
    uint64_t nfps = 0;
    /* Lookup hashes in the vqf filter (Random Lookup) */
    for (uint64_t i = 0; i < nvals; i++) {
      if (vqf_is_present(filter, other_vals[i])) {
        nfps++;
      }
    }
    gettimeofday(&end, &tzp);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    negative_throughput += 1.0 * nvals / elapsed_usecs;
      
    print_time_elapsed("Random lookup:", &start, &end, nvals, "random lookup");
    printf("%lu/%lu positives\nFP rate: 1/%f\n", nfps, nvals, 1.0 * nvals / nfps);

      
    gettimeofday(&start, &tzp);
    /* Delete hashes in the vqf filter */
    for (uint64_t i = 0; i < nvals; i++) {
      vqf_remove(filter, vals[i]);
      //bool success;
      //success = vqf_remove(filter, vals[i]);
      //if (success == false) {
      //  fprintf(stderr, "Remove failed for %ld\n", vals[i]);
      //  exit(EXIT_FAILURE);
      //}
    }
    gettimeofday(&end, &tzp);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    remove_throughput += 1.0 * nvals / elapsed_usecs;
      
      
    print_time_elapsed("Remove time", &start, &end, nvals, "remove");

    free(filter);
  }

  //printf("[TEST of %d]\n", TEST_NUM);
  //printf("Insertion throughput : %f Million operations / sec\n", 1.0 * insertion_throughput / TEST_NUM);
  //printf("Positive throughput  : %f Million operations / sec\n", 1.0 * positive_throughput / TEST_NUM);
  //printf("Negative throughput  : %f Million operations / sec\n", 1.0 * negative_throughput / TEST_NUM);
  //printf("Remove throughput    : %f Million operations / sec\n", 1.0 * remove_throughput / TEST_NUM);

  return 0;
}
