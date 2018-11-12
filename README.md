# Aula-7-Thrust
Este material é baseado na documentação disponível em https://github.com/thrust/thrust (originalmente de Jared Hoberock and Nathan Bell), https://docs.nvidia.com/cuda/thrust/index.html, no GPU Teaching Kit – Accelerated Computing e no livro  "Programming Massively Parallel Processors A Hands-on Approach" (3ra edição) de David B. Kirk e Wen-mei W. Hwu (leitura sugerida!!) e no Lab "Using Thrust to Accelerate C++", created by Mark Ebersole. É recomendável também fazer o laboratório em https://courses.nvidia.com/courses/course-v1:DLI+L-AC-18+V1/.

**Thrust** é uma biblioteca de algoritmos paralelos que se assemelha muito ao STL (C++ Standard Template Library), permitindo ao programador criar rapidamente programas portáveis que fazem uso tanto de GPUs quanto de arquiteturas multicore CPUs.  A interoperabilidade com tecnologias estabelecidas (como CUDA, TBB e OpenMP) facilita a integração com o software existente.

Para evitar conflitos de espaços de nome,  todas as funções e membros Thrust estarão precedidos por thrust:: para indicar de qual  namespace vêm. Também estaremos usando funções do namespace std:: 

### Containers ###
Enquanto o STL tem muitos tipos diferentes de conteiners, o Thrust trabalha apenas com dois tipos de vetores:
- Vetores armazenados no host são declarados com thrust :: host_vector <type>
- Vetores de dispositivo são declarados com thrust :: device_vector <type>

Ao declarar um vetor de host ou dispositivo, você deve fornecer o tipo de dados que ele conterá. Na verdade, como o Thrust é um modelo, a maioria das suas declarações envolverá a especificação de um tipo. Esses tipos podem ser tipos de dados nativos simples comuns, como int, char ou float. Mas o tipo também pode ser estruturas complexas como um thrust::tuple, que contém vários elementos. Para obter detalhes sobre como inicializar um vetor de host ou dispositivo, você pode consultar a documentação do Thrust em https://github.com/thrust/thrust/wiki/Documentation. Para este laboratório, os dois métodos necessários para inicializar um vetor Thrust são os seguintes:

- Criar um vetor de host ou dispositivo de um tamanho específico: thrust::host_vector <type> h_vec (SIZE); ou thrust :: device_vector <type> d_vec (SIZE); 
- Para criar e inicializar um vetor de dispositivo a partir de um vetor Thrust existente: thrust::device_vector <type> d_vec = h_vec;
   
Nos bastidores, o Thrust manipulará a alocação de espaço no dispositivo que tem o mesmo tamanho de h_vec, além de copiar a memória do host para o dispositivo.
### Exercício 1: Declare os vetores H e D e inicialíze-os com os valores indicados em cada linha em que aparece um #FIXME:
```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

int main(void)
{
    // Declare AQUI um vetor de host H para armazenar 4 inteiros 
#FIXME 
    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;
    
    // H.size() returns the size of vector H
    std::cout << "H has size " << H.size() << std::endl;

    // print contents of H
    for(int i = 0; i < H.size(); i++)
        std::cout << "H[" << i << "] = " << H[i] << std::endl;

    // resize H
    H.resize(2);
    
    std::cout << "H now has size " << H.size() << std::endl;

    // Declare AQUI um vetor de device D e inicialíze-o com os valores em H
#FIXME 
    
    // elements of D can be modified
    D[0] = 99;
    D[1] = 88;
    
    // print contents of D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    // H and D are automatically deleted when the function returns
    return 0;
}
```
Salve seu programa com extensão .cu e compile-o com o nvcc. Repare que elementos individuais de um *device_vector* podem ser acessados usando **[ ]**. Entretanto, já que esses acessos requerem chamadas a cudaMemcpy, devem ser usados com cuidado. Iremos estudar formas mais eficientes posteriormente. 

## Iteradores ##
Agora que temos containers para nossos dados no Thrust, precisamos que nossos algoritmos acessem esses dados independentemente do tipo de dados que eles contêm. É aqui que entram os iteradores de C++. No caso de containers vetoriais, que são realmente apenas matrizes, os iteradores podem ser considerados como ponteiros para elementos de matriz. Portanto, H.begin() é um iterador que aponta para o primeiro elemento da matriz armazenada dentro do vetor H. Da mesma forma, H.end() aponta para o elemento após o último elemento do vetor H.

Embora os iteradores vetoriais sejam semelhantes aos ponteiros, eles carregam mais informações com eles. Não precisamos dizer aos algoritmos de Thrust que eles estão operando em um iterador device_vector ou host_vector. Essa informação é capturada no tipo do iterador retornado pelo H.begin(). Quando uma função Thrust é chamada, ela inspeciona o tipo do iterador para determinar se deve usar uma implementação de host ou de dispositivo. Esse processo é conhecido como **static dispatching**, pois o dispatch do host/dispositivo é resolvido no momento da compilação. Observe que isso implica que não há sobrecarga de tempo de execução no processo de escalonamento.

## Funções ##
Com contêineres e iteradores, podemos finalmente processar nossos dados usando funções. Quase todas as funções Thrust processam os dados usando iteradores apontando para vetores diferentes. Por exemplo, para copiar dados de um vetor de dispositivo para um vetor de host, é usado o código a seguir:

```cpp
thrust :: copy (d_vec.begin (), d_vec.end (), h_vec.begin ());
```
Esta função simplesmente diz "Iniciando no primeiro elemento de d_vec, copie os dados iniciando no início de h_vec, avançando através de cada vetor até que o final de d_vec seja atingido."

O seguinte código computa a soma de 100 números randômicos em paralelo:
```cpp
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
```
### Exercício 2: Ordenação
Neste exercício você irá escrever código usando Thrust para copiar dados gerados aleatoriamente para a GPU, ordená-los e copiá-los de volta para o host. Seu objetivo nessa tarefa é substituir o #FIXME do programa a seguir pelo código que faz o seguinte:
1. Crie um device_vector e copie os dados inicializados em h_vec para um device_vetor d_vec usando o operador *=* como discutido acima
2. Ordene os dados no dispositivo com thrust::sort
3. Mova os dados de volta para h_vec usando thrust::copy

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
  // generate 50000000 random numbers serially
  #FIXME
  
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  #FIXME
  // sort data on the device (846M keys per second on GeForce GTX 480)
  #FIXME

  // transfer data back to host
  #FIXME
  return 0;
}
```

Conseguiu? Parabéns! Você executou código na GPU usando Thrust, sem precisar escrever código específico para a GPU. Daqui a pouco veremos como é possível, mudando só um switch do compilador switch, compilar Thrust para executar na CPU.

A maioria das funções Thrust são planejadas para serem blocos de construção, permitindo que o programador construa algoritmos complexos sobre elas. O objetivo desta tarefa é oferecer a você mais experiência usando funções e iteradores Thrust e expor você a funções adicionais disponíveis.

Além disso, você começará a trabalhar com "functores" nessa tarefa. Um functor é um "objeto de função", que é um objeto que pode ser chamado como se fosse uma função comum. Em C ++, um functor é apenas uma classe ou estrutura que define o operador de chamada de função. Por serem objetos, functores podem ser passados ​​(junto com seu estado) para outras funções como parâmetro. O empuxo vem com um punhado de functores predefinidos, um dos quais vamos usar nesta tarefa. Na próxima tarefa, veremos como escrever seu próprio functor e usá-lo em um algoritmo Thrust.

Existem algumas maneiras de usar um functor. Um deles é criá-lo como se fosse um objeto normal como este:

thrust :: modulus <float> modulusFunctor (...); // Crie o functor, se necessário, passe qualquer argumento para o construtor.
float result = modulusFunctor (4.0, 2.0); // Use o functor como uma função regular
...

O segundo método é chamar o construtor diretamente em uma lista de argumentos para outra função:

thrust :: transform (..., thrust :: modulus <float> ());

Você notará que temos que adicionar o () após <float> enquanto estamos chamando o construtor functors para instanciar o objeto da função. A função de transformação Thrust agora pode aplicar o functor a todos os elementos com os quais está trabalhando.

Usando o editor abaixo, abra task2.cu como antes (clique na pasta task2, depois em task2.cu). Seu objetivo é substituir as seções de código #FIXME para conseguir o seguinte. Observe que cada item é vinculado à documentação relevante do Thrust.

    Inicialize o vetor X com 0,1,2,3, ..., 9 usando thrust :: sequence
    Preencha o vetor Z com todos os 2 usando thrust :: fill
    Defina Y igual a X mod Z usando thrust :: transform e thrust :: modulus
    Substitua todos os 1's em Y por 10's com impulso :: substituir
    Imprima o resultado de Y com thrust :: copy e copie-o para o iterador de saída std :: ostream_iterator <int> (std :: cout, "\ n")

Para certificar-se de que você está recebendo a resposta correta, o programa imprime o vetor do dispositivo Y. Se tudo foi feito corretamente, você deverá ver a seguinte saída:

0
10
0
10
0
10
0
10
0
10

Tarefa 3

Thrust fornece alguns functores internos para você usar, mas o poder real vem da criação de seus próprios functores. Para essa tarefa, removeremos a chamada para thrust :: replace do código na Tarefa nº 2 e, em vez disso, substituiremos essa funcionalidade por um functor personalizado usado na chamada thrust :: transform. Um exemplo de um functor customizado é o seguinte functor unário que retorna o quadrado do valor de entrada:

template <typename T>
quadrado da estrutura
{
  __host__ __device__
  Operador T () (const T & x) const
  {
    return x * x;
  }
};

A linha __host__ __device__ acima diz ao compilador nvcc para compilar uma versão do Host e do Dispositivo da função abaixo dela. Isso mantém a portabilidade entre CPUs e GPUs.

A maneira como esse functor funciona é que estamos sobrescrevendo o operador () da estrutura. Este é o mais versátil dos operadores sobrecarregáveis, pois pode aceitar qualquer número e tipo de entradas e retornar qualquer tipo de saída. Dessa forma, os algoritmos Thrust precisam simplesmente chamar outputType = someFunctor (inputType1, inputType2, ..., inputTypeN) sem precisar entender o que a função faz. Isso contribui para uma biblioteca muito poderosa e flexível!

Nota: Não é necessário tornar seu functor personalizado um modelo struct, mas adiciona muita flexibilidade ao seu código.

Em task3.cu abaixo, conclua a criação do functor modZeroOrTen e, em seguida, chame-o a partir da função thrust :: transform. Se tudo for feito corretamente, você deve obter a mesma saída da Tarefa 2, que é:

0
10
0
10
0
10
0
10
0
10

Dica # 1
O functor personalizado para o nosso código precisa ser um operador binário - são necessários dois valores como entrada. O exemplo de functor quadrado mostrado é apenas um operador unário.

Dica # 2
Se você está criando o objeto functor quadrado diretamente na lista de argumentos thrust :: transform, não esqueça de adicionar o () para chamar o construtor.

Dica # 3
Não se esqueça de adicionar, no mínimo, a palavra-chave __device__ antes da sua função, para que o compilador saiba compilar esta função para a GPU.

Em [10]:

# Execute esta célula para compilar o task3.cu e, se for bem-sucedido, execute o programa

! nvcc -O2 -arch = sm_30 task3 / task3.cu -run

0
10
0
10
0
10
0
10
0
10

Criando esse functor personalizado, conseguimos eliminar a chamada thrust :: replace, o que contribui para uma aplicação mais eficiente.
Tarefa 4

Até agora, lidamos apenas com iteradores básicos que permitem ao Thrust percorrer todos os elementos de um vetor. Os iteradores extravagantes executam uma variedade de propósitos valiosos. Nesta tarefa, mostraremos como os iteradores sofisticados nos permitem atacar uma classe mais ampla de problemas com os algoritmos Thrust padrão. Apesar de não cobrirmos todos os iteradores de fantasia nesta tarefa, cobriremos três deles.

O mais simples do grupo, constant_iterator é simplesmente um iterador que retorna o mesmo valor sempre que o desreferimos. No exemplo a seguir, inicializamos um constant_iterator com o valor 10.

// criar iteradores
impulso :: constant_iterator primeiro (10);
thrust :: constant_iterator last = primeiro + 3; // Defina o último elemento como 3 depois do começo

// soma de [primeiro, último)
thrust :: reduce (primeiro, último); // retorna 30 (isto é, 10 + 10 + 10)

O transform_iterator nos permite aplicar a técnica de combinar algoritmos separados, sem precisar depender do Thrust para fornecer uma versão especial do algoritmo transform_xxx. Essa tarefa mostra outra maneira de fundir uma transformação com uma redução, desta vez com apenas redução simples aplicada a um transform_iterator.

O exemplo a seguir imprime todos os elementos no vetor de valores, depois de fixá-los entre 0 e 100.

thrust :: copy (thrust :: make_transform_iterator (values.begin (), clamp (0, 100)), // primeiro elemento
             thrust :: make_transform_iterator (values.end (), clamp (0, 100)), // elemento final
             std :: ostream_iterator (std :: cout, ""));

Finalmente, o zip_iterator é um gadget extremamente útil: ele toma múltiplas seqüências de entrada e produz uma sequência de tuplas. O exemplo a seguir aplica o arbitrary_functor a cada tupla, onde cada tupla é composta de elementos dos vetores A, B, C e D. Você pode ver detalhes sobre a função thrust :: for_each aqui.

thrust :: for_each (thrust :: make_zip_iterator (thrust :: make_tuple (A. begin (), B. begin (), C. begin (), D. begin ())),
                 thrust :: make_zip_iterator (thrust :: make_tuple (A.end (), B.end (), C.end (), D.end ())),
                 arbitrary_functor ());

Uma desvantagem de transform_iterator e zip_iterator é que pode ser complicado especificar o tipo completo do iterador, o que pode ser bastante demorado. Por esse motivo, é uma prática comum simplesmente colocar a chamada em make_transform_iterator ou make_zip_iterator nos argumentos do algoritmo que está sendo invocado.

Seu objetivo nesta tarefa é modificar task4.cu e escrever o código para implementar cada tipo de iterador. Os diferentes tipos de iteradores são divididos em três funções - não há necessidade de modificar a função main (). Se quiser, você pode comentar o interior das funções que você ainda precisa implementar enquanto se concentra em uma. 

## Counting iterator
´´´cpp
/*
Permite gerar uma sequência de valores crescentes. Nesse exemplo se inicializa um counting_iterator com o valor 10 e se acessa como se fosse um array.
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
  std::cout << thrust::reduce(first, last) << std::endl;   // returns 33 (= 10 + 11 + 12)
  
  return 0;
}
´´´
## Exercício: Histograma
The purpose of this lab is to implement a histogramming algorithm for an input array of integers. This approach composes several distinct algorithmic steps to compute a histogram, which makes Thrust a valuable tools for its implementation.
problem setup
Consider the dataset
input = [2 1 0 0 2 2 1 1 1 1 4]
A resulting histogram would be
histogram = [2 5 3 0 1]
reflecting 2 zeros, 5 ones, 3 twos, 0 threes, and one 4 in the input dataset. Note that the number of bins is equal to
max(input) + 1

histogram sort approach
First, sort the input data using thrust::sort. Continuing with the original example:
sorted = [0 0 1 1 1 1 1 2 2 2 4]
Determine the number of bins by inspecting the last element of the list and adding 1:
num_bins = sorted.back() + 1

To compute the histogram, we can compute the culumative histogram and then work backwards. To do this in Thrust, use thrust::upper_bound. upper_bound takes an input data range (the sorted input) and a set of search values, and for each search value will report the largest index in the input range that the search value could be inserted into without changing the sorted order of the inputs. For example,
[2 8 11 11 12] = thrust::upper_bound([0 0 1 1 1 1 1 2 2 2 4], // input [0 1 2 3 4]) // search
By carefully crafting the search data, thrust::upper_bound will produce a cumulative histogram. The search data must be a range [0,num_bins).
Once the cumulative histogram is produced, use thrust::adjacent_different to compute the histogram.
[2 5 3 0 1] = thrust::adjacent_difference([2 8 11 11 12])
Check the thrust documentation for details of how to use upper_bound and adjacent_difference. Instead of constructing the search array in device memory, you may be able to use thrust::counting_iterator.

instructions
Edit the code in the code tab to perform the following:
• allocate space for input on the GPU • copy host memory to device
• invoke thrust functions
• copy results from device to host
Instructions about where to place each part of the code is demarcated by the //@@ comment lines.

## Instruções
The executable generated as a result of compiling the lab can be run using the following command:
./ThrustHistogramSort_Template -e <expected.raw> \ -i <input.raw> -o <output.raw> -t integral_vector
2
where <expected.raw> is the expected output, <input.raw> is the input dataset, and <output.raw> is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.
attribution
This is a simplified version of the material presented in the Thrust repository aqui: https://github.com/thrust/thrust/blob/master/examples/histogram.cu

code template
The following code is suggested as a starting point for students. The code handles the import and export as well as the checking of the solution. Stu- dents are expected to insert their code is the sections demarcated with //@@. Students expected the other code unchanged. 

```cpp
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength, num_bins;
  unsigned int *hostInput, *hostBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  // Copy the input to the GPU
  wbTime_start(GPU, "Allocating GPU memory");
  //@@ Insert code here
  wbTime_stop(GPU, "Allocating GPU memory");

  // Determine the number of bins (num_bins) and create space on the host
  //@@ insert code here
  num_bins = deviceInput.back() + 1;
  hostBins = (unsigned int *)malloc(num_bins * sizeof(unsigned int));

  // Allocate a device vector for the appropriate number of bins
  //@@ insert code here

  // Create a cumulative histogram. Use thrust::counting_iterator and
  // thrust::upper_bound
  //@@ Insert code here

  // Use thrust::adjacent_difference to turn the culumative histogram
  // into a histogram.
  //@@ insert code here.

  // Copy the histogram to the host
  //@@ insert code here

  // Check the solution is correct
  wbSolution(args, hostBins, num_bins);

  // Free space on the host
  //@@ insert code here
  free(hostBins);

  return 0;
}
```

## Trabalho para casa ##
Você encontrará no diretório  /usr/local/cuda/cuda9-installed-samples/NVIDIA_CUDA-9.0_Samples/6_Advanced/radixSortThrust uma implementação de Radix Sort em paralelo usando Thrust. Comente o arquivo .cu e apresente-o na próxima aula. O trabalho pode ser feito em duplas. 

