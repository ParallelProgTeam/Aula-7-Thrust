# Aula-7-Thrust
Este material é baseado na documentação disponível em https://github.com/thrust/thrust (originalmente de Jared Hoberock and Nathan Bell), no GPU Teaching Kit – Accelerated Computing e no livro  "Programming Massively Parallel Processors A Hands-on Approach" (3ra edição) de David B. Kirk e Wen-mei W. Hwu (leitura sugerida!!) e no Lab "Using Thrust to Accelerate C++", created by Mark Ebersole. É recomendável também fazer o laboratório em https://courses.nvidia.com/courses/course-v1:DLI+L-AC-18+V1/.

**Thrust** é uma biblioteca de algoritmos paralelos que se assemelha muito ao STL (C++ Standard Template Library), permitindo ao programador criar rapidamente programas portáveis que fazem uso tanto de GPUs quanto de arquiteturas multicore CPUs.  A interoperabilidade com tecnologias estabelecidas (como CUDA, TBB e OpenMP) facilita a integração com o software existente.

Por exemplo, o seguinte código 

 The code in Task #1 will move some randomly generated data to the GPU, sort it, and then copy it back to the host. Before we jump into the code, let's go over the basics of working with Thrust.

First off, to help avoid naming conflicts, C++ makes use of namepsaces and Thrust is no exception. In this lab and in most of the Thrust code you will see, all the Thrust functions and members will be preceded by thrust:: to indicate which namespace it comes from. In this this lab, you will also see reference to the std:: namespace for printing out values to the screen.
Containers

Whereas the STL has many different types of containers, Thrust just works with two vector types:

    Host vectors are declared with thrust::host_vector<type>
    Device vectors are declared with thrust::device_vector<type>

When declaring a host or device vector, you must provide the data type it will contain. In fact, since Thrust is a template library, most of your declarations will involve specifying a type. These types can be common simple native data-types like int, char, or float. But the type can also be complex structures like a thrust::tuple which contains multiple elements. For details on how to initialize a host or device vector, I encourage you to look at the Thrust documentation here. For this lab, the two methods needed to initialize a Thrust vector are the following:

    Create a host or device vector of a specific size: thrust::host_vector<type> h_vec( SIZE ); or thrust::device_vector<type> d_vec( SIZE );
        It's common practice to proceed host vector variables with h_ and device vector variables with d_ to make it clear in the code which memory space they are referring to.
    Create and initialize a device vector from an existing Thrust vector: thrust::device_vector<type> d_vec = h_vec;
        Under the covers, Thrust will handle allocating space on the device that is the same size as h_vec, as well as copying the memory from the host to the device.

Interators

Now that we have containers for our data in Thrust, we need a way for our algorithms to access this data, regardless of what type of data they contain. This is where C++ iterators come in to play. In the case of vector containers, which are really just arrays, iterators can be thought of as pointers to array elements. Therefore, H.begin() is an iterator that points to the first element of the array stored inside the H vector. Similarly, H.end() points to the element one past the last element of the H vector.

Although vector iterators are similar to pointers, they carry more information with them. We do not have to tell Thrust algorithms that they are operating on a device_vector or host_vector iterator. This information is captured in the type of the iterator returned by H.begin(). When a Thrust function is called, it inspects the type of the iterator to determine whether to use a host or a device implementation. This process is known as static dispatching since the host/device dispatch is resolved at compile time. Note that this implies that there is no runtime overhead to the dispatch process.
Functions

With containers and iterators, we can finally process our data using functions. Almost all Thrust functions process the data by using iterators pointing at different vectors. For example, to copy data from a device vector to a host vector, the following code is used:

thrust::copy( d_vec.begin(), d_vec.end(), h_vec.begin() );

This function simply states "Starting at the first element of d_vec, copy the data starting at the beginning of h_vec, advancing through each vector until the end of d_vec is reached."
Task Instructions

Your objective in this task is to replace the #FIXME of task1.cu with code that does the following:

    Create a device_vector and copy the initialized h_vec data to it using the = operator as discussed above
    Sort the data on the device with thrust::sort
    Move the data back to h_vec using thrust::copy

The solution to this task is provided in task1_solution.cu in the editor below. Please look at it to check your work, or if you get stuck. You can find this file by clicking on the "task1" folder on the left of the text editor, then selecting task1_solution.cu.

After making a change, make sure to save the file by simply clicking the save button below. As a reminder, saving the file actually saves it on the Amazon GPU system in the cloud you're running on. To get a copy of the files we'll be working on, consult the Post-Lab section near the end of this page. Also remember to keep an eye on the time. The instance you are running on will shut down after 120 minutes from when you started the lab, so make sure to save your work before times run out!



Thrust is best explained through examples. The following source code generates random numbers serially and then transfers them to a parallel device where they are sorted.
```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

int main(void)
{
  // generate 32M random numbers serially
  thrust::host_vector<int> h_vec(32 << 20);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<int> d_vec = h_vec;

  // sort data on the device (846M keys per second on GeForce GTX 480)
  thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  return 0;
}
```

Congrats! You have successfully executed code on the GPU using Thrust, and you did not have to write any GPU specific code! As we'll see in Task #4, with just a compiler switch, you can compile Thrust code to execute on a CPU.












This code sample computes the sum of 100 random numbers in parallel:

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstdlib>

int main(void)
{
  // generate random data serially
  thrust::host_vector<int> h_vec(100);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device and compute sum
  thrust::device_vector<int> d_vec = h_vec;
  int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
  return 0;
}

