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
Salve seu programa com extensão .cu e compile-o com o 
```bash
nvcc -O2 -arch = sm_30 task3 / task3.cu -run
```
Repare que elementos individuais de um *device_vector* podem ser acessados usando **[ ]**. Entretanto, já que esses acessos requerem chamadas a cudaMemcpy, devem ser usados com cuidado. Iremos estudar formas mais eficientes posteriormente. 

## Iteradores ##
Agora que temos containers para nossos dados no Thrust, precisamos que nossos algoritmos acessem esses dados independentemente do tipo de dados que eles contêm. É aqui que entram os iteradores de C++. No caso de containers vetoriais, que são realmente apenas matrizes, os iteradores podem ser considerados como ponteiros para elementos de matriz. Portanto, H.begin() é um iterador que aponta para o primeiro elemento da matriz armazenada dentro do vetor H. Da mesma forma, H.end() aponta para o elemento após o último elemento do vetor H.

Embora os iteradores vetoriais sejam semelhantes aos ponteiros, eles carregam mais informações com eles. Não precisamos dizer aos algoritmos de Thrust que eles estão operando em um iterador device_vector ou host_vector. Essa informação é capturada no tipo do iterador retornado pelo H.begin(). Quando uma função Thrust é chamada, ela inspeciona o tipo do iterador para determinar se deve usar uma implementação de host ou de dispositivo. Esse processo é conhecido como **static dispatching**, pois o dispatch do host/dispositivo é resolvido no momento da compilação. Observe que isso implica que não há sobrecarga de tempo de execução no processo de escalonamento.

## Funções ##
Com conteiners e iteradores, podemos finalmente processar nossos dados usando funções. Quase todas as funções Thrust processam os dados usando iteradores apontando para vetores diferentes. Por exemplo, para copiar dados de um vetor de dispositivo para um vetor de host, é usado o código a seguir:

```cpp
thrust::copy (d_vec.begin (), d_vec.end (), h_vec.begin ());
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

A maioria das funções Thrust são planejadas para serem *building blocks* (blocos de construção), permitindo que o programador construa algoritmos complexos sobre elas. O objetivo desta tarefa é oferecer a você mais experiência usando funções e iteradores Thrust e expor você a funções adicionais disponíveis.

<!--Além disso, você começará a trabalhar com "functors" nessa tarefa. -->
Um functor é um "objeto de função", que é um objeto que pode ser chamado como se fosse uma função comum. Em C++, um functor é apenas uma classe ou estrutura que define o operador de chamada de função. Por serem objetos, functores podem ser passados (junto com seu estado) para outras funções como parâmetro. Thrust vem com um punhado de functores predefinidos, um dos quais vamos usar nesta tarefa. Na próxima tarefa, veremos como escrever seu próprio functor e usá-lo em um algoritmo Thrust.

Existem algumas maneiras de usar um functor. Um deles é criá-lo como se fosse um objeto normal como este:
´´´cpp
thrust :: modulus <float> modulusFunctor (...); // Crie o functor, se necessário, passe qualquer argumento para o construtor.
float result = modulusFunctor (4.0, 2.0); // Use o functor como uma função regular
...
´´´
O segundo método é chamar o construtor diretamente em uma lista de argumentos para outra função:
´´´cpp
thrust :: transform (..., thrust :: modulus <float> ());
´´´
Você notará que temos que adicionar o () após <float> enquanto estamos chamando o construtor functors para instanciar o objeto da função. A função de transformação Thrust agora pode aplicar o functor a todos os elementos com os quais está trabalhando.

<!--
### Exercício 3
Nesse exercício, você deve substituir as seções de código #FIXME para resolver os seguintes problemas:
*   Inicialize o vetor X com 0,1,2,3, ..., 9 usando thrust::sequence
*    Preencha o vetor Z com todos os 2 usando thrust::fill
*    Defina Y igual a X mod Z usando thrust::transform e thrust::modulus
*    Substitua todos os 1's em Y por 10's com thrust::replace
*    Imprima o resultado de Y com thrust::copy e copie-o para o iterador de saída std::ostream_iterator<int>(std::cout, "\n")
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
Thrust fornece alguns functores internos para você usar, mas o poder real vem da criação de seus próprios functores. Para essa tarefa, removeremos a chamada para thrust::replace do código na Tarefa nº 3 e, em vez disso, substituiremos essa funcionalidade por um functor personalizado usado na chamada thrust::transform. Um exemplo de um functor customizado é o seguinte functor unário que retorna o quadrado do valor de entrada:
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
No exercício abaixo, conclua a criação do functor modZeroOrTen e, em seguida, chame-o a partir da função thrust::transform. Se tudo for feito corretamente, você deve obter a mesma saída do exercício anterior, que é:
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
*Dica # 1: O functor personalizado para o nosso código precisa ser um operador binário - são necessários dois valores como entrada. O exemplo de functor quadrado mostrado é apenas um operador unário.
*Dica # 2: Se você está criando o objeto functor quadrado diretamente na lista de argumentos thrust :: transform, não esqueça de adicionar o () para chamar o construtor.
*Dica # 3: Não se esqueça de adicionar, no mínimo, a palavra-chave __device__ antes da sua função, para que o compilador saiba compilar esta função para a GPU.
Criando esse functor personalizado, conseguimos eliminar a chamada thrust :: replace, o que contribui para uma aplicação mais eficiente.
-->

### Exemplo: Um bom motivo para usarmos functors:
```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cstdlib>

int main(void)
{
  // generate some random data on the host
	thrust::host_vector<int> h_vec(10);
	for (unsigned int i=0;i<h_vec.size();i++) h_vec[i]=rand()%10;

  // transfer to device
	thrust::device_vector<int> d_vec = h_vec;

  // sum on device
	int final_sum = thrust::reduce(d_vec.begin(), d_vec.end(), 
		0, thrust::plus<int>());
	int final_max = thrust::reduce(d_vec.begin(), d_vec.end(), 
		0, thrust::maximum<int>());
	int final_min = thrust::reduce(d_vec.begin(), d_vec.end(), 
		999, thrust::minimum<int>());

	std::cout<<"Final sum="<<final_sum<<"  max="<<final_max<<"  min="<<final_min<<"\n";

	return 0;
}
```
Não é muito eficiente chamar thrust::reduce três vezes no mesmo vetor. É mais eficiente chamá-lo uma vez e coletar a soma, min e max de uma só vez. Para fazer isso, precisamos escrever um 'functor' esquisito para passar para thrust::reduce.

```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cstdlib>

class sum_min_max {
public:
	int sum, min, max;
	sum_min_max() {sum=0; min=1000000000; max=-1000000000;}
	__device__ __host__ sum_min_max(int value) {sum=value; min=value; max=value;}
};

// This 'functor' function object combines two sum_min_max objects
class smm_combiner {
public:
__device__ __host__ 
sum_min_max operator()(sum_min_max l,const sum_min_max &r) {
	l.sum+=r.sum;
	if (l.min>r.min) l.min=r.min;
	if (l.max<r.max) l.max=r.max;
	return l;
}
};

int main(void)
{
  // generate some random data on the host
	thrust::host_vector<int> h_vec(10);
	for (unsigned int i=0;i<h_vec.size();i++) h_vec[i]=rand()%10;

  // transfer to device
	thrust::device_vector<int> d_vec = h_vec;

  // sum/min/max on device
	sum_min_max final = thrust::reduce(d_vec.begin(), d_vec.end(), 
		sum_min_max(), smm_combiner());

	std::cout<<"Final sum="<<final.sum<<"  max="<<final.max<<"  min="<<final.min<<"\n";

	return 0;
}

//This same idea could probably be better written as a thrust::tuple.
```

## Exemplo: 
Até agora, lidamos apenas com iteradores básicos que permitem ao Thrust percorrer todos os elementos de um vetor. Nesta tarefa, mostraremos como os iteradores sofisticados nos permitem atacar uma classe mais ampla de problemas que com os algoritmos Thrust padrão. 
O mais simples do grupo, **constant_iterator**, é simplesmente um iterador que retorna o mesmo valor sempre que o desreferimos. Ele permite gerar uma sequência de valores crescentes. Nesse exemplo se inicializa um counting_iterator com o valor 10 e se acessa como se fosse um array.

## Counting iterator
´´´cpp
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

<!-- O **transform_iterator** nos permite aplicar a técnica de combinar algoritmos separados, sem precisar depender do Thrust para fornecer uma versão especial do algoritmo transform_xxx. Essa tarefa mostra outra maneira de fundir uma transformação com uma redução, desta vez com apenas redução simples aplicada a um transform_iterator.
O exemplo a seguir imprime todos os elementos no vetor de valores, depois de fixá-los entre 0 e 100.
thrust :: copy (thrust :: make_transform_iterator (values.begin (), clamp (0, 100)), // primeiro elemento
             thrust :: make_transform_iterator (values.end (), clamp (0, 100)), // elemento final
             std :: ostream_iterator (std :: cout, ""));
Finalmente, o zip_iterator é um gadget extremamente útil: ele toma múltiplas seqüências de entrada e produz uma sequência de tuplas. O exemplo a seguir aplica o arbitrary_functor a cada tupla, onde cada tupla é composta de elementos dos vetores A, B, C e D. Você pode ver detalhes sobre a função thrust :: for_each aqui.
thrust :: for_each (thrust :: make_zip_iterator (thrust :: make_tuple (A. begin (), B. begin (), C. begin (), D. begin ())),
                 thrust :: make_zip_iterator (thrust :: make_tuple (A.end (), B.end (), C.end (), D.end ())),
                 arbitrary_functor ());
Uma desvantagem de transform_iterator e zip_iterator é que pode ser complicado especificar o tipo completo do iterador, o que pode ser bastante demorado. Por esse motivo, é uma prática comum simplesmente colocar a chamada em make_transform_iterator ou make_zip_iterator nos argumentos do algoritmo que está sendo invocado.
Seu objetivo nesta tarefa é modificar task4.cu e escrever o código para implementar cada tipo de iterador. Os diferentes tipos de iteradores são divididos em três funções - não há necessidade de modificar a função main (). Se quiser, você pode comentar o interior das funções que você ainda precisa implementar enquanto se concentra em uma. -->

*upper_bound* é uma versão vetorizada de uma busca binária: para cada iterador v em [values_first, values_last) tenta encontrar o valor  *v em um intervalo ordenado  [first, last). Returna o índice da última posaição onde o valor poderia ser inserido sem violar a ordenação.
Parameters
    first	The beginning of the ordered sequence.
    last	The end of the ordered sequence.
    values_first	The beginning of the search values sequence.
    values_last	The end of the search values sequence.
    result	The beginning of the output sequence.
Template Parameters
    ForwardIterator	is a model of Forward Iterator.
    InputIterator	is a model of Input Iterator. and InputIterator's value_type is LessThanComparable.
    OutputIterator	is a model of Output Iterator. and ForwardIterator's difference_type is convertible to OutputIterator's value_type.
Precondition
    The ranges [first,last) and [result, result + (last - first)) shall not overlap.
Exemplo:
´´´cpp
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
...
thrust::device_vector<int> input(5);
input[0] = 0;
input[1] = 2;
input[2] = 5;
input[3] = 7;
input[4] = 8;
thrust::device_vector<int> values(6);
values[0] = 0; 
values[1] = 1;
values[2] = 2;
values[3] = 3;
values[4] = 8;
values[5] = 9;
thrust::device_vector<unsigned int> output(6);
thrust::upper_bound(input.begin(), input.end(),
                    values.begin(), values.end(),
                    output.begin());
// output is now [1, 1, 2, 2, 5, 5]
´´´
_host__ __device__ OutputIterator thrust::adjacent_difference 	( const thrust::detail::execution_policy_base< DerivedPolicy > &  	exec,
		InputIterator  	first,
		InputIterator  	last,
		OutputIterator  	result 
	) 		
´´´
	
adjacent_difference calcula as diferenças dos elementos adjacentes no intervalo [first, last]. Ou seja, \*first é atribuído a \*result e, para cada iterador i no intervalo [first + 1, last], a diferença de \*i e \*(i - 1) é designada para \*(result + (i - first)).

O trecho de código a seguir demonstra como usar adjacent_difference para calcular a diferença entre elementos adjacentes de um intervalo usando a política de execução thrust::device:

´´´cpp
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
...
int h_data[8] = {1, 2, 1, 2, 1, 2, 1, 2};
thrust::device_vector<int> d_data(h_data, h_data + 8);
thrust::device_vector<int> d_result(8);
thrust::adjacent_difference(thrust::device, d_data.begin(), d_data.end(), d_result.begin());
// d_result is now [1, 1, -1, 1, -1, 1, -1, 1]
´´´
Veja mais informação aqui: https://developer.download.nvidia.com/CUDA/training/introductiontothrust.pdf
e nos exemplos no Moodle.

## Exercício: Histograma
O objetivo deste laboratório é implementar um algoritmo de histograma para uma matriz de entrada de inteiros. Essa abordagem compõe várias etapas algorítmicas distintas para calcular um histograma, o que torna o Thrust uma ferramenta valiosa para sua implementação.
Considere o conjunto de dados:
input = [2 1 0 0 2 2 1 1 1 1 4]
Um histograma resultante seria
histograma = [2 5 3 0 1]
refletindo 2 zeros, 5 uns, 3 dois, 0 três e um 4 no dataset de entrada. Observe que o número de compartimentos é igual a
max (entrada) + 1

### Abordagem de ordenação do histograma
Primeiro, classifique os dados de entrada usando thrust :: sort. Continuando com o exemplo original:
ordenado = [0 0 1 1 1 1 1 2 2 2 4]
Determine o número de bins inspecionando o último elemento da lista e adicionando 1:
num_bins = sorted.back () + 1

Para calcular o histograma, podemos calcular o histograma culomativo e depois retroceder. Para fazer isso no Thrust, use thrust::upper_bound. *upper_bound* recebe um intervalo de dados de entrada (a entrada classificada) e um conjunto de valores de pesquisa e, para cada valor de pesquisa, reportará o maior índice no intervalo de entrada no qual o valor da pesquisa poderia ser inserido sem alterar a ordem classificada das entradas. Por exemplo,
[2 8 11 11 12] = thrust::upper_bound ([0 0 1 1 1 1 1 2 2 2 4], // entrada [0 1 2 3 4]) // pesquisa
Criando cuidadosamente os dados de pesquisa, thrust::upper_bound produzirá um histograma cumulativo. Os dados de pesquisa devem ser um intervalo [0, num_bins).
Uma vez que o histograma cumulativo é produzido, use thrust :: adjacent_different para calcular o histograma.
[2 5 3 0 1] = thrust::adjacent_difference ([2 8 11 11 12])
Verifique a documentação de empuxo para obter detalhes sobre como usar upper_bound e adjacent_difference. Em vez de construir a matriz de pesquisa na memória do dispositivo, você poderá usar thrust::counting_iterator.

## Instruções
O código a seguir é sugerido como ponto de partida. Insira seu código nas seções demarcadas com #FIXME. O restante do código deve permanecer inalterado. 

```cpp
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iomanip>
#include <iterator>

// This example illustrates a method for computing a
// histogram [1] with Thrust.  We consider standard "dense"
// histograms, where some bins may have zero entries
// For example, histograms for the data set
//    [2 1 0 0 2 2 1 1 1 1 4]
// which contains 2 zeros, 5 ones, and 3 twos and 1 four, is
//    [2 5 3 0 1]
// using the dense method 
//
// The best histogramming methods depends on the application.
// If the number of bins is relatively small compared to the 
// input size, then the binary search-based dense histogram
// method is probably best.  If the number of bins is comparable
// to the input size, then the reduce_by_key-based sparse method 
// ought to be faster.  When in doubt, try both and see which
// is fastest.
//
// [1] http://en.wikipedia.org/wiki/Histogram


// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

// dense histogram using binary search
template <typename Vector1, 
          typename Vector2>
void dense_histogram(const Vector1& input,
                           Vector2& histogram)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // criar device_vector data e copiar dados de input em data (pode ignorar se o input pode ser destruído)
  #FIXME  
  
  // print the initial data
  print_vector("initial data", data);

  // ordenar o device_vector data para juntar os elementos iguais
  #FIXME
      
  // print the sorted data
  print_vector("sorted data", data);

  // number of histogram bins is equal to the maximum value plus one
  IndexType num_bins = data.back() + 1;

  // resize histogram storage
  histogram.resize(num_bins);
  
  // encontrar o final de cada bin de valores: (dica: use thrust::upper_bound)
    thrust::counting_iterator<IndexType> search_begin(0);
    #FIXME
  
  // print the cumulative histogram
  print_vector("cumulative histogram", histogram);

  // computar o histograma calculando as diferenças no histograma cumulativo (dica: use thrust::adjacent_difference)
  #FIXME
    
  // print the histogram
  print_vector("histogram", histogram);
}

int main(void)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 9);

  const int N = 40;
  const int S = 4;

  // generate random data on the host
  thrust::host_vector<int> input(N);
  for(int i = 0; i < N; i++)
  {
    int sum = 0;
    for (int j = 0; j < S; j++)
      sum += dist(rng);
    input[i] = sum / S;
  }

  // demonstrate dense histogram method
  {
    std::cout << "Dense Histogram" << std::endl;
    thrust::device_vector<int> histogram;
    dense_histogram(input, histogram);
  }

  return 0;
}

Histogram
          initial data  3 4 3 5 8 5 6 6 4 4 5 3 2 5 6 3 1 3 2 3 6 5 3 3 3 2 4 2 3 3 2 5 5 5 8 2 5 6 6 3 
           sorted data  1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 8 8 
  cumulative histogram  0 1 7 19 23 32 38 38 40 
             histogram  0 1 6 12 4 9 6 0 2 
```

## Trabalho para casa ##
Você encontrará no diretório  /usr/local/cuda/cuda9-installed-samples/NVIDIA_CUDA-9.0_Samples/6_Advanced/radixSortThrust uma implementação de Radix Sort em paralelo usando Thrust. Comente o arquivo .cu e apresente-o na última aula. O trabalho pode ser feito em duplas. 

