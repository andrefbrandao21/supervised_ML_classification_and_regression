# Regressão linear com múltiplos recursos

## Introdução à regressão linear com múltiplos recursos

### Regressão linear com múltiplas características

Originalmente, a regressão linear usava uma única característica (ex.: tamanho da casa) para prever o preço. Agora, com múltiplas características (ex.: tamanho da casa, número de quartos, andares, idade da casa), o modelo se torna mais informativo.

### Notação para múltiplos recursos

- **$X_1, X_2, X_3, X_4$** representam as quatro características.
- **$X_j$** representa a lista de recursos, onde **$j$** varia de 1 a 4.
- **$n$** é o número total de características (neste caso, **$n = 4$**).
- **$X^{(i)}$** representa o vetor de características do **$i$-ésimo** exemplo de treinamento.
- **$X^{(i)}_j$** representa o valor da **$j$-ésima** característica no **$i$-ésimo** exemplo de treinamento.

### Modelo de regressão linear com múltiplos recursos

O modelo é definido como:

$$
f_{w,b}(X) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b
$$

**Exemplo concreto**: previsão do preço da casa com base em tamanho, quartos, andares e idade.

**Interpretação dos parâmetros**:
- **$b$**: preço base da casa.
- **$w_1, w_2, w_3, w_4$**: pesos que indicam a contribuição de cada característica para o preço.

### Notação vetorial

- **$W$** é um vetor de parâmetros **$w_1, w_2, \dots, w_n$**.
- **$X$** é um vetor de características **$x_1, x_2, \dots, x_n$**.
- O modelo pode ser reescrito como:

$$
f_{w,b}(X) = W \cdot X + b
$$

onde **$\cdot$** representa o produto escalar (dot product).

### Produto escalar (dot product)

O produto escalar de dois vetores **$W$** e **$X$** é calculado como:

$$
W_1X_1 + W_2X_2 + \dots + W_nX_n
$$

Essa notação simplifica a expressão do modelo.

### Regressão linear múltipla

- Modelo de regressão linear com múltiplos recursos de entrada.
- Contrasta com a regressão univariada, que usa apenas uma característica.
- O termo "regressão multivariada" refere-se a a outro conceito não abordado aqui.

## Vetorização

Nesta seção, exploramos uma ideia muito útil chamada **vetorização**. Quando você está implementando um algoritmo de aprendizado, o uso da vetorização torna seu código mais curto e também faz com que ele seja executado com muito mais eficiência. Aprender a escrever código vetorizado permite aproveitar bibliotecas modernas de álgebra linear e até mesmo hardware de GPU (Unidade de Processamento Gráfico), que é projetado para acelerar computações gráficas, mas também pode ser usado para executar código vetorizado de forma mais rápida.

### Exemplo concreto de vetorização

Considere um exemplo com os parâmetros **$w$** e **$b$**, onde **$w$** é um vetor com três números e **$x$** é um vetor de características também com três números. Aqui, **$n = 3$**. Na álgebra linear, a indexação começa em 1, então o primeiro valor é **$w_1$** e **$x_1$**. No código Python, você pode definir essas variáveis usando a biblioteca **NumPy**, que é amplamente utilizada para álgebra linear numérica.

```python
import numpy as np

w = np.array([w1, w2, w3])
b = 5
x = np.array([x1, x2, x3])
```


Em Python, a indexação começa em 0, então você acessa os valores de **$w$** e **$x$** da seguinte forma:

- **$w[0]$**: primeiro valor de **$w$**.
- **$w[1]$**: segundo valor de **$w$**.
- **$w[2]$**: terceiro valor de **$w$**.

O mesmo vale para **$x$**.

### Implementação sem vetorização

Sem vetorização, você pode calcular a previsão do modelo multiplicando cada parâmetro **$w$** pelo recurso associado **$x$** e somando os resultados. Isso pode ser feito com um loop **for**:

```python
f = 0
for j in range(n):
    f += w[j] * x[j]
f += b
```

No entanto, essa abordagem é ineficiente, especialmente quando **$n$** é grande (ex.: 100 ou 100.000).

### Implementação com vetorização

Com vetorização, a mesma operação pode ser feita de forma muito mais eficiente usando o produto escalar (**dot product**) da biblioteca NumPy:

```python
f = np.dot(w, x) + b
```

A função **`np.dot`** realiza o produto escalar entre os vetores **$w$** e **$x$**, que é matematicamente definido como:

$$
f = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

### Benefícios da vetorização

1. **Código mais curto**: A implementação vetorizada é concisa e fácil de ler.
2. **Execução mais rápida**: A função **`np.dot`** utiliza hardware paralelo (CPU ou GPU) para acelerar a computação, tornando-a muito mais eficiente do que loops ou cálculos sequenciais.

### O que acontece nos bastidores?

A vetorização permite que operações matemáticas sejam executadas em paralelo, aproveitando ao máximo o hardware disponível. Isso é especialmente útil quando **$n$** é grande, pois evita a necessidade de escrever manualmente cada termo da soma.

### Explicação do Código com e sem Vetorização

O código apresentado compara duas abordagens para atualizar os parâmetros **$w$** em um algoritmo de aprendizado de máquina: uma **sem vetorização** (usando loops) e outra **com vetorização** (usando operações matriciais).

---

#### **1. Código sem Vetorização**

Aqui, cada parâmetro **$w_j$** é atualizado individualmente, um por um, usando um loop **for**. Isso significa que o computador executa cada operação de forma sequencial.

**Matematicamente**:

$$
w_1 = w_1 - 0.1 \cdot d_1
$$

$$
w_2 = w_2 - 0.1 \cdot d_2
$

$$
\vdots
$$

$$ w_{16} = w_{16} - 0.1 \cdot d_{16}$$

**Implementação em Python**:
```python
for j in range(0, 16):
    w[j] = w[j] - 0.1 * d[j]
```

**Funcionamento**:
- O loop **for** itera sobre cada índice **$j$** de 0 a 15.
- Para cada **$j$**, o valor de **$w[j]$** é atualizado subtraindo **$0.1 \cdot d[j]$**.
- Isso resulta em 16 operações sequenciais, uma para cada **$w_j$**.

**Desvantagens**:
- **Ineficiente**: O computador realiza cada operação uma após a outra, o que é lento, especialmente para grandes conjuntos de dados.
- **Código mais longo**: Requer mais linhas de código para realizar a mesma tarefa.

---

#### **2. Código com Vetorização**

Aqui, a atualização dos parâmetros **$w$** é feita de forma vetorizada, ou seja, todas as operações são realizadas simultaneamente usando operações matriciais.

**Matematicamente**:
$$
\vec{w} = \vec{w} - 0.1 \cdot \vec{d}
$$

**Implementação em Python**:
```python
w = w - 0.1 * d
```

**Funcionamento**:
- **$\vec{w}$** e **$\vec{d}$** são vetores (ou arrays) que contêm todos os valores de **$w_j$** e **$d_j$**, respectivamente.
- A operação **$w = w - 0.1 \cdot d$** é realizada em uma única linha de código.
- Nos bastidores, o computador usa hardware paralelo para realizar todas as subtrações e multiplicações ao mesmo tempo.

**Vantagens**:
- **Eficiência**: As operações são realizadas em paralelo, o que é muito mais rápido do que a execução sequencial.
- **Código mais curto**: A implementação é concisa e fácil de ler.
- **Escalabilidade**: Funciona bem mesmo com grandes conjuntos de dados (ex.: milhares de parâmetros).

---

## Resumo dos conceitos aprendidos

- **Revisão dos conceitos aprendidos**:
  - Gradiente descendente.
  - Regressão linear múltipla.
  - Vetorização.

- **Implementação do gradiente descendente para regressão linear múltipla com vetorização**:
  - Uso de notação vetorial para simplificar a representação dos parâmetros.
  - Parâmetros **$w_1$** a **$w_n$** são agrupados em um vetor **$\vec{w}$**.
  - O modelo é representado como **$f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$**, onde **$\cdot$** é o produto escalar.

- **Função de custo**:
  - A função de custo **$J$** é definida como uma função do vetor **$\vec{w}$** e do número **$b$**.
  - **$J(\vec{w}, b)$** retorna um número que representa o custo.

- **Atualização dos parâmetros no gradiente descendente**:
  - Cada parâmetro **$w_j$** é atualizado como **$w_j = w_j - \alpha \cdot \frac{\partial J}{\partial w_j}$**.
  - O parâmetro **$b$** é atualizado como **$b = b - \alpha \cdot \frac{\partial J}{\partial b}$**.
  - A derivada parcial **$\frac{\partial J}{\partial w_j}$** é calculada considerando o erro de previsão.

- **Diferenças entre regressão univariada e múltipla**:
  - Na regressão univariada, há apenas uma característica **$x$**.
  - Na regressão múltipla, há **$n$** características, e cada **$w_j$** é atualizado individualmente.

- **Método da equação normal**:
  - Alternativa ao gradiente descendente para regressão linear.
  - Resolve **$\vec{w}$** e **$b$** diretamente sem iterações.
  - Desvantagens:
    - Não é generalizável para outros algoritmos de aprendizado.
    - Lento para um grande número de características.
  - Usado em bibliotecas de aprendizado de máquina, mas não é comum implementá-lo manualmente.

# Gradiente descendente na prática
## Técnicas para Melhorar o Desempenho do Gradiente Descendente

### Dimensionamento de Recursos
- **Objetivo**: Acelerar a execução do gradiente descendente.
- **Exemplo**: Prever o preço de uma casa com duas características:
  - $x_1$: Tamanho da casa (300 a 2.000 pés quadrados).
  - $x_2$: Número de quartos (0 a 5 quartos).

### Relação entre Tamanho do Recurso e Parâmetros
- **Exemplo de Parâmetros**:
  - $w_1 = 50$, $w_2 = 0.1$, $b = 50$.
  - Preço estimado: $100.000 + 0.5k + 50k = 100.5k$ (muito longe de $500k$).
  
- **Outro Exemplo de Parâmetros**:
  - $w_1 = 0.1$, $w_2 = 50$, $b = 50$.
  - Preço estimado: $0.1 \times 2000 + 50 \times 5 + 50 = 200k + 250k + 50 = 500k$ (estimativa correta).

### Impacto do Tamanho dos Recursos nos Parâmetros
- **Recursos com Faixa Grande**: Parâmetros menores (ex.: $w_1 = 0.1$).
- **Recursos com Faixa Pequena**: Parâmetros maiores (ex.: $w_2 = 50$).

### Gráfico de Dispersão e Função de Custo
- **Gráfico de Dispersão**:
  - Eixo horizontal ($x_1$): Tamanho da casa (faixa maior).
  - Eixo vertical ($x_2$): Número de quartos (faixa menor).

- **Gráfico de Contorno da Função de Custo**:
  - Contornos ovais/elípticos devido à diferença de escala entre $x_1$ e $x_2$.
  - Pequenas mudanças em $w_1$ têm grande impacto no custo $J$.
  - Grandes mudanças em $w_2$ são necessárias para alterar significativamente o custo $J$.

### Problema com Gradiente Descendente sem Escalonamento
- **Oscilações**: Gradiente descendente pode oscilar por muito tempo antes de convergir.
- **Causa**: Contornos altos e estreitos devido à diferença de escala entre $x_1$ e $x_2$.

### Solução: Escalonamento de Recursos
- **Transformação dos Dados**:
  - Redimensionar $x_1$ e $x_2$ para faixas comparáveis (ex.: 0 a 1).
- **Resultado**:
  - Contornos mais circulares.
  - Gradiente descendente converge mais rapidamente ao mínimo global.


## Métodos de Escalonamento

### Divisão pelo Máximo
- **Exemplo para $x_1$** (tamanho da casa):
  - Faixa original: 300 a 2000.
  - Escalonado: $x_1^{\text{escalado}} = \frac{x_1}{2000}$.
  - Nova faixa: 0.15 a 1.

- **Exemplo para $x_2$** (número de quartos):
  - Faixa original: 0 a 5.
  - Escalonado: $x_2^{\text{escalado}} = \frac{x_2}{5}$.
  - Nova faixa: 0 a 1.

### Normalização Média
- **Passos**:
  1. Calcular a média ($\mu_1$) do recurso $x_1$.
  2. Subtrair a média e dividir pela diferença entre máximo e mínimo.
  
- **Exemplo para $x_1$**:
  - Média ($\mu_1$): 600.
  - Escalonado: $x_1^{\text{normalizado}} = \frac{x_1 - \mu_1}{2000 - 300}$.
  - Nova faixa: -0.18 a 0.82.

- **Exemplo para $x_2$**:
  - Média ($\mu_2$): 2.3.
  - Escalonado: $x_2^{\text{normalizado}} = \frac{x_2 - \mu_2}{5 - 0}$.
  - Nova faixa: -0.46 a 0.54.

### Normalização do Escore Z
- **Passos**:
  1. Calcular a média ($\mu$) e o desvio padrão ($\sigma$) do recurso.
  2. Subtrair a média e dividir pelo desvio padrão.

- **Exemplo para $x_1$**:
  - Média ($\mu_1$): 600.
  - Desvio padrão ($\sigma_1$): 450.
  - Escalonado: $x_1^{\text{Z-score}} = \frac{x_1 - \mu_1}{\sigma_1}$.
  - Nova faixa: -0.67 a 3.1.

- **Exemplo para $x_2$**:
  - Média ($\mu_2$): 2.3.
  - Desvio padrão ($\sigma_2$): 1.4.
  - Escalonado: $x_2^{\text{Z-score}} = \frac{x_2 - \mu_2}{\sigma_2}$.
  - Nova faixa: -1.6 a 1.9.

### Regras Gerais para Escalonamento
- **Faixa Ideal**: -1 a 1 (ou valores próximos).
- **Exceções**:
  - Se um recurso varia entre -3 e 3, ou -0.3 e 0.3, ainda é aceitável.
  - Recursos com valores muito grandes (ex.: -100 a 100) ou muito pequenos (ex.: -0.001 a 0.001) devem ser escalonados.

### Benefícios do Escalonamento
- **Acelera a convergência** do gradiente descendente.
- **Facilita a visualização** dos dados e da função de custo.

## Método de Verificação de convergência do gradiente descendente

### Curva de Aprendizado
- **Definição**: Gráfico que mostra o valor da função de custo $J(w, b)$ após cada iteração do gradiente descendente.
- **Eixo Horizontal**: Número de iterações do gradiente descendente.
- **Eixo Vertical**: Valor da função de custo $J(w, b)$.

### Comportamento Esperado
- **Custo Decrescente**: Se o gradiente descendente estiver funcionando corretamente, $J(w, b)$ deve diminuir após cada iteração.
- **Estabilização**: Após um certo número de iterações, $J(w, b)$ pode se estabilizar, indicando que o gradiente descendente convergiu.

### Problemas Comuns
- **Aumento do Custo**: Se $J(w, b)$ aumentar após uma iteração, pode indicar:
  - Taxa de aprendizado $\alpha$ muito alta.
  - Bug no código.

### Número de Iterações
- **Variação**: O número de iterações necessárias para convergir pode variar significativamente entre diferentes aplicações:
  - Pode ser tão pouco quanto 30 iterações.
  - Pode exigir até 100.000 iterações.

### Teste Automático de Convergência
- **Definição**: Declarar convergência se a redução no custo $J(w, b)$ for menor que um valor pequeno $\epsilon$ (ex.: 0.001) em uma iteração.
- **Desafio**: Escolher o valor adequado para $\epsilon$ pode ser difícil.

### Recomendação
- **Visualização Gráfica**: Observar a curva de aprendizado é geralmente mais informativo do que confiar em testes automáticos de convergência.
- **Inspeção Visual**: Ajuda a identificar problemas no funcionamento do gradiente descendente.


## Escolha da Taxa de Aprendizado (Alpha) no Gradiente Descendente

### Importância da Taxa de Aprendizado
- **Taxa de Aprendizado Muito Pequena**: O algoritmo converge muito lentamente.
- **Taxa de Aprendizado Muito Grande**: O algoritmo pode não convergir, oscilando ou divergindo.

### Sinais de Problemas com a Taxa de Aprendizado
- **Custo Oscilante**: Se o custo $J(w, b)$ aumenta e diminui alternadamente, isso pode indicar:
  - **Taxa de Aprendizado Muito Grande**: Atualizações excessivas fazem o algoritmo "pular" o mínimo.
  - **Bug no Código**: Verificar a implementação do gradiente descendente.

- **Custo Aumentando Consistentemente**: Se o custo $J(w, b)$ aumenta após cada iteração, isso geralmente indica:
  - **Taxa de Aprendizado Muito Grande**: Atualizações são tão grandes que o algoritmo se afasta do mínimo.
  - **Erro no Código**: Verificar se o sinal negativo está presente na atualização dos parâmetros:
    - Correto: $w_1 = w_1 - \alpha \times \frac{\partial J}{\partial w_1}$.
    - Incorreto: $w_1 = w_1 + \alpha \times \frac{\partial J}{\partial w_1}$.

### Depuração e Escolha de Alpha
- **Teste com Alpha Pequeno**:
  - Definir $\alpha$ como um valor muito pequeno (ex.: 0.0001).
  - Verificar se o custo $J(w, b)$ diminui a cada iteração.
  - Se o custo ainda oscilar ou aumentar, pode haver um bug no código.

- **Desvantagem de Alpha Pequeno**:
  - O algoritmo pode levar muitas iterações para convergir.

### Escolha Prática de Alpha
- **Testar uma Faixa de Valores**:
  - Começar com valores pequenos (ex.: 0.001) e aumentar gradualmente (ex.: 0.003, 0.01, 0.03, 0.1).
  - Para cada valor de $\alpha$, executar o gradiente descendente por algumas iterações e plotar o custo $J(w, b)$.
  - Escolher o maior valor de $\alpha$ que ainda faz o custo diminuir de forma consistente.

- **Estratégia de Multiplicação por 3**:
  - Testar valores de $\alpha$ multiplicando por 3 (ex.: 0.001, 0.003, 0.01, 0.03, 0.1).
  - Encontrar o valor ideal que equilibra velocidade de convergência e estabilidade.

