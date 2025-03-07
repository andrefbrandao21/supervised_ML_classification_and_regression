# Notas de Aula
**Autor:** André Filipe  
**Data:** 05/03/2025

---

## Sumário

- [Introdução](#introdução)
- [Visão Geral](#visão-geral)
  - [Importância do Aprendizado de Máquina](#importância-do-aprendizado-de-máquina)
  - [Aplicações Reais de ML](#aplicações-reais-de-ml)
  - [Impacto do ML em Diversos Setores](#impacto-do-ml-em-diversos-setores)
  - [Demanda por Habilidades em ML](#demanda-por-habilidades-em-ml)
- [Aprendizado Supervisionado e Não Supervisionado](#aprendizado-supervisionado-e-não-supervisionado)
  - [Aprendizado Supervisionado](#aprendizado-supervisionado)
  - [Aprendizado Não Supervisionado](#aprendizado-não-supervisionado)
- [Regressão](#regressão)
  - [Regressão Linear](#regressão-linear)
  - [Função Custo](#função-custo)
    - [Aplicabilidade da Função de Custo em Regressão](#aplicabilidade-da-função-de-custo-em-regressão)
      - [Treinamento do Modelo](#treinamento-do-modelo)
        - [Avaliação do Modelo](#avaliação-do-modelo)
        - [Comparação de Modelos](#comparação-de-modelos)
        - [Exemplo Prático](#exemplo-prático)
        - [Exemplo Matemático](#exemplo-matemático)
- [Gradiente Descendente](#gradiente-descendente)
  - [Introdução ao Gradiente Descendente](#introdução-ao-gradiente-descendente)
  - [Implementação do Gradiente Descendente](#implementação-do-gradiente-descendente)
    - [Atualização dos Parâmetros](#atualização-dos-parâmetros)
    - [Taxa de Aprendizado](#taxa-de-aprendizado)
    - [Atualização Simultânea](#atualização-simultânea)
    - [Convergência](#convergência)
    - [Detalhes sobre Derivadas](#detalhes-sobre-derivadas)
    - [Intuição sobre o Gradiente Descendente](#intuição-sobre-o-gradiente-descendente)
      - [Taxa de Aprendizado](#taxa-de-aprendizado-α)
      - [Derivada e Intuição](#derivada-e-intuição)
      - [Exemplos de Comportamento](#exemplos-de-comportamento)
    - [Gradiente Descendente para Regressão Linear](#gradiente-descendente-para-regressão-linear)
      - [Modelo de Regressão Linear](#modelo-de-regressão-linear)
      - [Função de Custo](#função-de-custo-erro-quadrático-médio)
      - [Gradiente Descendente](#gradiente-descendente)
      - [Derivadas Parciais](#derivadas-parciais)
      - [Propriedades da Função de Custo](#propriedades-da-função-de-custo)
---

## Introdução
Estas são as notas de aula da primeira semana do curso de Aprendizado de Máquina - Stanford.

---

## Visão Geral

- **Importância do Aprendizado de Máquina:**
  - Cresceu como um subcampo da IA para construir máquinas inteligentes.
  - Soluciona problemas complexos que não podem ser resolvidos com programação explícita (ex.: reconhecimento de fala, diagnóstico médico, carros autônomos).

- **Aplicações Reais de ML:**
  - Exemplos de aplicações em grandes empresas (Google Brain, Baidu, landing.AI):
    - Reconhecimento de fala.
    - Visão computacional (Google Maps, Street View).
    - Combate a fraudes de pagamento.
    - Carros autônomos.
    - Agricultura, saúde, comércio eletrônico e manufatura.

- **Impacto do ML em Diversos Setores:**
  - ML já cria valor significativo na indústria de software.
  - Oportunidades inexploradas em setores como varejo, transporte, manufatura, etc.
  - Estima-se que ML e IA gerem US\$ 13 trilhões anualmente até 2030 (McKinsey).

- **Demanda por Habilidades em ML:**
  - Grande demanda por profissionais qualificados em ML.
  - Momento ideal para aprender e dominar essas habilidades.

---

## Aprendizado Supervisionado e Não Supervisionado

### Aprendizado Supervisionado

- Algoritmos que aprendem mapeamentos de entrada ($x$) para saída ($y$), com base em exemplos fornecidos (dados rotulados).
- Fornece ao algoritmo exemplos de entradas ($x$) e saídas corretas ($y$), permitindo que ele aprenda a prever $y$ a partir de $x$.
- Exemplos de aplicações:
  - **Filtro de Spam:** Entrada ($x$) = e-mail; Saída ($y$) = spam ou não spam.
  - **Reconhecimento de Fala:** Entrada ($x$) = clipe de áudio; Saída ($y$) = transcrição do texto.
  - **Tradução Automática:** Entrada ($x$) = texto em inglês; Saída ($y$) = tradução em outro idioma.
  - **Publicidade Online:** Entrada ($x$) = informações sobre o anúncio e o usuário; Saída ($y$) = probabilidade de o usuário clicar no anúncio.
  - **Carros Autônomos:** Entrada ($x$) = imagem e dados de sensores; Saída ($y$) = posição de outros carros.
  - **Inspeção Visual na Manufatura:** Entrada ($x$) = imagem de um produto; Saída ($y$) = presença de defeitos.

- Funcionamento do Aprendizado Supervisionado:
  - **Treinamento:** O modelo é treinado com pares de entrada ($x$) e saída correta ($y$).
  - **Previsão:** Após o treinamento, o modelo pode prever a saída ($y$) para novas entradas ($x$) que nunca viu antes.

- Tipos de aprendizado supervisionado:
  - **Classificação:** Saída é uma variável discreta (ex.: spam ou não spam).
  - **Regressão:** Saída é uma variável contínua (ex.: preço de uma casa).

### Aprendizado Não Supervisionado

- Dados não estão associados a rótulos de saída ($y$).
- Objetivo: Encontrar estrutura, padrões ou algo interessante nos dados.
- Agrupa dados em clusters (grupos) com base em similaridades.

---

## Regressão

### Regressão Linear

- Modelo mais simples de regressão.
- Ajusta uma linha reta aos dados.
- No contexto de machine learning, a notação correta pode ser visualizada na tabela abaixo:

| Notação Geral       | Descrição                                                                 | Python (se aplicável) |
|---------------------|---------------------------------------------------------------------------|-----------------------|
| $a$                 | Escalar, não negrito.                                                     | -                     |
| $\mathbf{a}$        | Vetor, negrito.                                                           | -                     |
| $\mathbf{x}$        | Valores das características dos exemplos de treinamento (neste caso, tamanho em 1000 pés quadrados). | `x_train`             |
| $\mathbf{y}$        | Valores alvo dos exemplos de treinamento (neste caso, preço em milhares de dólares). | `y_train`             |
| $x^{(i)}, y^{(i)}$  | $i$-ésimo exemplo de treinamento.                                         | `x_i`, `y_i`          |
| $m$                 | Número de exemplos de treinamento.                                        | `m`                   |
| $w$                 | Parâmetro: peso.                                                          | `w`                   |
| $b$                 | Parâmetro: viés (bias).                                                   | `b`                   |
| $f_{w,b}(x^{(i)})$  | Resultado da avaliação do modelo em $x^{(i)}$ parametrizado por $w, b$: $f_{w,b}(x^{(i)}) = wx^{(i)}+b$. | `f_wb`                |

### Função Custo
- Serve como uma métrica para avaliar o quão bem um modelo está performando ao comparar suas previsões com os valores reais. A intuição por trás da função de custo pode ser entendida como uma forma de "medir o erro" do modelo, ou seja, quantificar o quão longe as previsões estão dos valores verdadeiros.

#### Aplicabilidade da Função de Custo em Regressão

##### Treinamento do Modelo

- Durante o treinamento, o objetivo é ajustar os parâmetros do modelo ($\theta$) para minimizar a função de custo.
- Por exemplo, em uma regressão linear, os parâmetros são os coeficientes da reta:
  $$
  y = \theta_0 + \theta_1 x
  $$
- A minimização da função de custo garante que a reta se ajuste o melhor possível aos dados.

##### Avaliação do Modelo

- A função de custo também é usada para avaliar o desempenho do modelo em dados não vistos (conjunto de teste).
- Um valor baixo da função de custo indica que o modelo está fazendo previsões precisas.

##### Comparação de Modelos

- Diferentes modelos (ou diferentes configurações de um mesmo modelo) podem ser comparados com base no valor da função de custo.
- O modelo com o menor custo é geralmente o melhor.

##### Exemplo Prático

- Suponha que você está tentando prever o preço de casas com base no tamanho ($x$) e no número de quartos.
- O modelo faz previsões ($\hat{y}$) para cada casa, e a função de custo (MSE) calcula o erro médio entre as previsões e os preços reais.
- Durante o treinamento, o **Gradiente Descendente** ajusta os parâmetros ($\theta$) para minimizar o MSE, resultando em um modelo que faz previsões mais precisas.

##### Exemplo Matemático

Considere um conjunto de dados com $m$ exemplos. A função de custo (MSE) é dada por:

$$
  J(w, b) = \frac{1}{2m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2
$$

onde

$$f_{w,b}(x^{(i)}) = wx^{(i)} + b $$
- $f_{w,b}(x^{(i)})$ é nossa previsão para o exemplo $i$ usando os parâmetros $w,b$.  
- $(f_{w,b}(x^{(i)}) -y^{(i)})^2$ é a diferença quadrada entre o valor alvo e a previsão.  
- Essas diferenças são somadas para todos os $m$ exemplos e divididas por `2m` para produzir o custo, $J(w,b)$.

## Gradiente Descendente

### Introdução ao Gradiente Descendente:
   - O gradiente descendente é um algoritmo usado para minimizar funções de custo, como $J(w, b)$, em modelos de aprendizado de máquina.
   - É amplamente utilizado, não apenas em regressão linear, mas também em modelos avançados, como redes neurais (aprendizado profundo).
   - Encontrar os valores de $w$ e $b$ que minimizam a função de custo $J(w, b)$.
   - O algoritmo começa com valores iniciais para $w$ e $b$ (geralmente 0) e ajusta esses parâmetros iterativamente para reduzir o custo.

   - O gradiente descendente ajusta $w$ e $b$ em pequenos passos, sempre na direção que reduz mais rapidamente o custo $J(w, b)$.
   - A direção escolhida é a da **descida mais íngreme**, que é calculada matematicamente.

---

### Implementação do Gradiente Descendente

#### Atualização dos Parâmetros:
   - O gradiente descendente atualiza os parâmetros $w$ e $b$ iterativamente para minimizar a função de custo $J(w, b)$.
   - A fórmula de atualização para $w$ é:
    $
     w = w -
     \alpha \cdot \frac{d}{dw} J(w, b)
    $
   - A fórmula de atualização para $b$ é:
     $
     b = b - \alpha \cdot \frac{d}{db} J(w, b)
     $
   - Onde:
     - $\alpha$: Taxa de aprendizado (controla o tamanho do passo).
     - $\frac{d}{dw} J(w, b)$ e $\frac{d}{db} J(w, b)$: Derivadas parciais da função de custo em relação a $w$ e $b$, respectivamente.

#### Taxa de Aprendizado ($\alpha$):
   - $\alpha$ é um pequeno número positivo (por exemplo, 0,01) que controla o tamanho dos passos na direção da descida.
   - Se $\alpha$ for muito grande, o algoritmo pode "pular" o mínimo.
   - Se $\alpha$ for muito pequeno, o algoritmo pode demorar muito para convergir.

#### Atualização Simultânea:
   - É crucial atualizar $w$ e $b$ **simultaneamente**.
   - A implementação correta envolve:
     - Calcular os novos valores de $w$ e $b$ em variáveis temporárias (`temp_w` e `temp_b`).
     - Atualizar $w$ e $b$ com os valores temporários.
     - A atualização simultânea é feita da seguinte forma:
     
     - $$
     \begin{aligned}
     temp\_w &= w - \alpha \frac{\partial}{\partial w} J(w, b) \\
     temp\_b &= b - \alpha \frac{\partial}{\partial b} J(w, b) \\
     w &= temp\_w \\
     b &= temp\_b
     \end{aligned}
     $$

   - A atualização não simultânea pode levar a resultados incorretos, pois o valor de $w$ ou $b$ já atualizado afeta o cálculo da derivada. A atualização não simultânea é feita da seguinte maneira:
     $$
     \begin{aligned}
     temp\_w &= w - \alpha \frac{\partial}{\partial w} J(w, b) \\
     w &= temp\_w \\
     temp\_b &= b - \alpha \frac{\partial}{\partial b} J(w, b) \\
     b &= temp\_b
     \end{aligned}
     $$

#### Convergência:
   - O algoritmo converge quando $w$ e $b$ param de mudar significativamente a cada iteração.
   - Nesse ponto, o algoritmo atinge um mínimo local da função de custo.

#### Detalhes sobre Derivadas:
   - As derivadas $\frac{d}{dw} J(w, b)$ e $\frac{d}{db} J(w, b)$ indicam a direção da descida mais íngreme.
   - Mesmo sem conhecimento de cálculo, é possível entender e implementar o gradiente descendente com base nas fórmulas fornecidas.

---

### Intuição sobre o Gradiente Descendente

#### Taxa de Aprendizado ($\alpha$):
   - Controla o tamanho dos passos na direção da descida.
   - Se $\alpha$ for muito pequeno, o algoritmo pode demorar para convergir.
   - Se $\alpha$ for muito grande, o algoritmo pode "pular" o mínimo.
   - A escolha de $\alpha$ é crucial para o desempenho do algoritmo.

#### Derivada e Intuição:
   - A derivada $\frac{d}{dw} J(w)$ representa a inclinação da função de custo no ponto atual.
   - Se a derivada for **positiva**, $w$ é reduzido (movendo-se para a esquerda no gráfico).
   - Se a derivada for **negativa**, $w$ é aumentado (movendo-se para a direita no gráfico).
   - Em ambos os casos, o objetivo é se aproximar do mínimo da função de custo.

#### Exemplos de Comportamento:
   - **Exemplo 1**: Inicialização de $w$ à direita do mínimo.
     - A derivada é positiva, então $w$ é reduzido, movendo-se para a esquerda em direção ao mínimo.
   - **Exemplo 2**: Inicialização de $w$ à esquerda do mínimo.
     - A derivada é negativa, então $w$ é aumentado, movendo-se para a direita em direção ao mínimo.

---

### Gradiente Descendente para Regressão Linear

#### Modelo de Regressão Linear:
   - O modelo é definido por:
     $$
     f_{w,b}(x) = wx + b
     $$
   - Onde:
     - $w$: Coeficiente angular (peso).
     - $b$: Intercepto (viés).

#### Função de Custo (Erro Quadrático Médio):
   - A função de custo $J(w, b)$ é dada por:
     $$
     J(w, b) = \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2
     $$
   - Onde:
     - $m$: Número de exemplos de treinamento.
     - $f_{w,b}(x^{(i)})$: Previsão do modelo para o exemplo $i$.
     - $y^{(i)}$: Valor real para o exemplo $i$.

#### Derivadas Parciais:
   - A derivada em relação a $w$ é:
     $$
     \frac{\partial}{\partial w} J(w, b) = \frac{1}{m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}
     $$
   - A derivada em relação a $b$ é:
     $$
     \frac{\partial}{\partial b} J(w, b) = \frac{1}{m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})
     $$

#### Propriedades da Função de Custo:
   - A função de custo de erro quadrático médio é **convexa**, o que significa que ela tem um único mínimo global.
   - Isso garante que o gradiente descendente sempre convergirá para o mínimo global, desde que a taxa de aprendizado $\alpha$ seja escolhida adequadamente.