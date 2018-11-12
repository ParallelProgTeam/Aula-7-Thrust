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
Salve seu programa com extensão .cu e compile-o com o nvcc. Repare que elementos individuais de um *device_vector* podem ser acessados usando **[]**. Entretanto, já que esses acessos requerem chamadas a cudaMemcpy, devem ser usados com cuidado. Iremos estudar formas mais eficientes posteriormente. 

## Iteradores ##
Agora que temos containers para nossos dados no Thrust, precisamos que nossos algoritmos acessem esses dados independentemente do tipo de dados que eles contêm. É aqui que entram os iteradores de C++. No caso de containers vetoriais, que são realmente apenas matrizes, os iteradores podem ser considerados como ponteiros para elementos de matriz. Portanto, H.begin() é um iterador que aponta para o primeiro elemento da matriz armazenada dentro do vetor H. Da mesma forma, H.end() aponta para o elemento após o último elemento do vetor H.

Embora os iteradores vetoriais sejam semelhantes aos ponteiros, eles carregam mais informações com eles. Não precisamos dizer aos algoritmos de Thrust que eles estão operando em um iterador device_vector ou host_vector. Essa informação é capturada no tipo do iterador retornado pelo H.begin(). Quando uma função Thrust é chamada, ela inspeciona o tipo do iterador para determinar se deve usar uma implementação de host ou de dispositivo. Esse processo é conhecido como **static dispatching**, pois o dispatch do host/dispositivo é resolvido no momento da compilação. Observe que isso implica que não há sobrecarga de tempo de execução no processo de escalonamento.

## Funções ##
Com contêineres e iteradores, podemos finalmente processar nossos dados usando funções. Quase todas as funções Thrust processam os dados usando iteradores apontando para vetores diferentes. Por exemplo, para copiar dados de um vetor de dispositivo para um vetor de host, é usado o código a seguir:

```cpp
thrust :: copy (d_vec.begin (), d_vec.end (), h_vec.begin ());
```
Esta função simplesmente diz "Iniciando no primeiro elemento de d_vec, copie os dados iniciando no início de h_vec, avançando através de cada vetor até que o final de d_vec seja atingido."
### Exercício : Ordenação
Neste exercício você irá escrever código usando Thrust para copiar dados gerados aleatoriamente para a GPU, ordená-los e copiá-los de volta para o host. 

Seu objetivo nessa tarefa é substituir o #FIXME de task1.cu pelo código que faz o seguinte:

    Crie um device_vector e copie os dados h_vec inicializados para ele usando o operador = como discutido acima
    Classifique os dados no dispositivo com thrust :: sort
    Mova os dados de volta para h_vec usando thrust :: copy

A solução para essa tarefa é fornecida em task1_solution.cu no editor abaixo. Por favor, olhe para ele para verificar o seu trabalho, ou se você ficar preso. Você pode encontrar esse arquivo clicando na pasta "task1" à esquerda do editor de texto e selecionando task1_solution.cu.

Depois de fazer uma alteração, salve o arquivo simplesmente clicando no botão Salvar abaixo. Como um lembrete, salvar o arquivo realmente o salva no sistema GPU da Amazon na nuvem em que você está rodando. Para obter uma cópia dos arquivos em que trabalharemos, consulte a seção Post-Lab no final desta página. Lembre-se também de ficar de olho no tempo. A instância em que você está executando será encerrada após 120 minutos do início do laboratório. Portanto, salve seu trabalho antes que o tempo acabe!


### Exercício : Ordenação
Neste exercício você irá escrever código usando Thrust para copiar dados gerados aleatoriamente para a GPU, ordená-los e copiá-los de volta para o host. 

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

## Trabalho para casa ##
Você encontrará no diretório  /usr/local/cuda/cuda9-installed-samples/NVIDIA_CUDA-9.0_Samples/6_Advanced/radixSortThrust uma implementação de Radix Sort em paralelo usando Thrust. Comente o arquivo .cu e apresente-o na próxima aula. O trabalho pode ser feito em duplas. 

