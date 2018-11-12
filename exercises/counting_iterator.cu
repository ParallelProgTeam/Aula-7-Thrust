/*
4.2. counting_iterator
If a sequence of increasing values is required, then counting_iterator is the appropriate choice. Here we initialize a counting_iterator with the value 10 and access it like an array.
*/
#include <iostream>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

int main() {
// create iterators
  thrust::counting_iterator<int> first(10);
  thrust::counting_iterator<int> last = first + 3;

  std::cout << first[0] << " " ;  // returns 10
  std::cout << first[1] << " ";  // returns 11
  std::cout << first[100] <<  std::endl; // returns 110

// sum of [first, last)
  std::cout << thrust::reduce(first, last) << std::endl;   // returns 33 (i.e. 10 + 11 + 12)
  
  return 0;
}


