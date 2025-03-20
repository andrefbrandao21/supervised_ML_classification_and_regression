# Classificação com Regressão Logística

## Motivações

### Exemplos de Problemas de Classificação
- Identificar se um e-mail é spam (resposta: sim ou não).
- Detectar transações financeiras fraudulentas online.
- Classificar tumores como malignos ou benignos.

### Classificação Binária
- Problemas com apenas duas saídas possíveis (ex.: 0 ou 1, falso ou verdadeiro, negativo ou positivo).
- Convenções de nomenclatura:
  - Classe negativa: 0, falso, ausência (ex.: e-mail não é spam).
  - Classe positiva: 1, verdadeiro, presença (ex.: e-mail é spam).
- A escolha entre positivo e negativo é arbitrária e pode variar dependendo do contexto.

### Limitações da Regressão Linear em Classificação
- A regressão linear pode prever valores fora do intervalo 0 e 1, o que não é ideal para classificação.
- Exemplo prático: classificação de tumores malignos vs. benignos.
- Adicionar um ponto extremo ao conjunto de dados pode deslocar a linha de regressão, prejudicando a precisão da classificação.

### Introdução à Regressão Logística
- Algoritmo projetado para classificação binária, com saídas sempre entre 0 e 1.
- Evita os problemas da regressão linear em problemas de classificação.
- Apesar do nome "regressão", é usado para classificação.

## Regressão Logística

### Problemas com Regressão Linear em Classificação
- Regressão linear não é adequada para problemas de classificação.
- Em vez de uma linha reta, a regressão logística ajusta uma curva em forma de "S" aos dados.

### Função Sigmoid (Logística)
- Função matemática central na regressão logística.
- Fórmula: $$g(z) = \frac{1}{1 + e^{-z}}$$, onde $e$ é uma constante matemática (~2.7).
- Características da função Sigmoid:
  - Saída sempre entre 0 e 1.
  - Quando $z$ é grande, $g(z)$ se aproxima de 1.
  - Quando $z$ é muito negativo, $g(z)$ se aproxima de 0.
  - Quando $z = 0$, $g(z) = 0.5$.

### Construção do Modelo de Regressão Logística
1. **Passo 1**: Calcular $$z = w \cdot x + b$$, onde:
   - $w$: pesos.
   - $x$: características de entrada.
   - $b$: viés (bias).
2. **Passo 2**: Aplicar a função Sigmoid a $z$:
   - $$f(x) = g(z) = \frac{1}{1 + e^{-z}}$$.
   - Saída: valor entre 0 e 1, representando a probabilidade de $y = 1$.

### Interpretação da Saída da Regressão Logística
- A saída $f(x)$ representa a probabilidade de $y = 1$ dado $x$.
- Exemplo: Se $f(x) = 0.7$, há 70% de chance de $y = 1$ (tumor maligno) e 30% de chance de $y = 0$ (tumor benigno).
- Notação matemática: $$f(x) = P(y = 1 | x; w, b)$$, onde $w$ e $b$ são parâmetros do modelo.

### Aplicações e Importância
- Amplamente utilizado em publicidade online para decidir quais anúncios exibir.
- Base de muitos sistemas de classificação em larga escala.

## Resumo sobre Fronteira de Decisão em Regressão Logística

### Previsões com Limiar (Threshold)
- Para prever se $y = 0$ ou $y = 1$, define-se um limiar (comumente 0.5):
  - Se $f(x) \geq 0.5$, prevê-se $\hat{y} = 1$.
  - Se $f(x) < 0.5$, prevê-se $\hat{y} = 0$.
- Condição para $f(x) \geq 0.5$: $z \geq 0$, ou seja, $w \cdot x + b \geq 0$.

### Fronteira de Decisão
- A **fronteira de decisão** é a linha onde $w \cdot x + b = 0$.
  - Para $w \cdot x + b \geq 0$, o modelo prevê $y = 1$.
  - Para $w \cdot x + b < 0$, o modelo prevê $y = 0$.
- Exemplo com duas características ($x_1$ e $x_2$):
  - Parâmetros: $w_1 = 1$, $w_2 = 1$, $b = -3$.
  - Fronteira de decisão: $x_1 + x_2 - 3 = 0$ (linha reta $x_1 + x_2 = 3$).
    - À direita da linha: prevê $y = 1$.
    - À esquerda da linha: prevê $y = 0$.

### Fronteiras de Decisão Não Lineares
- Com **características polinomiais**, a fronteira de decisão pode ser não linear.
  - Exemplo: $$z = w_1 x_1^2 + w_2 x_2^2 + b$$.
  - Com $w_1 = 1$, $w_2 = 1$, $b = -1$, a fronteira de decisão é $x_1^2 + x_2^2 = 1$ (um círculo).
    - Fora do círculo: prevê $y = 1$.
    - Dentro do círculo: prevê $y = 0$.
- Com polinômios de ordem superior, é possível obter fronteiras de decisão complexas (ex.: elipses, formas irregulares).

### Limitações e Flexibilidade
- Sem características polinomiais, a fronteira de decisão é sempre linear (uma linha reta).
- Com polinômios de alta ordem, a regressão logística pode se ajustar a dados complexos.

# Função Custo

## Resumo sobre Função de Custo em Regressão Logística

### Introdução
- A **função de custo** mede o quão bem um conjunto de parâmetros se ajusta aos dados de treinamento.
- Para regressão logística, a função de custo de erro quadrático (usada em regressão linear) não é ideal, pois resulta em uma função não convexa, com múltiplos mínimos locais.

## Problema com o Erro Quadrático
- Em regressão linear, a função de custo é convexa (forma de tigela), permitindo que o gradiente descendente encontre o mínimo global.
- Em regressão logística, se usarmos o erro quadrático, a função de custo se torna não convexa, com muitos mínimos locais, dificultando a convergência do gradiente descendente.

### Nova Função de Custo para Regressão Logística
- Para garantir convexidade, uma nova função de custo é definida com base em uma **função de perda** específica.
- A função de perda para um único exemplo de treinamento é definida como:
  - Se $y = 1$: $$L(f(x), y) = -\log(f(x))$$.
  - Se $y = 0$: $$L(f(x), y) = -\log(1 - f(x))$$.

### Intuição da Função de Perda
- Para $y = 1$:
  - Se $f(x)$ (a previsão) estiver próxima de 1, a perda é próxima de 0 (previsão correta).
  - Se $f(x)$ estiver próxima de 0, a perda tende ao infinito (penaliza previsões muito erradas).
- Para $y = 0$:
  - Se $f(x)$ estiver próxima de 0, a perda é próxima de 0 (previsão correta).
  - Se $f(x)$ estiver próxima de 1, a perda tende ao infinito (penaliza previsões muito erradas).

### Função de Custo Total
- A função de custo $J(w, b)$ é a média das funções de perda sobre todos os exemplos de treinamento:
  $$
  J(w, b) = \frac{1}{m} \sum_{i=1}^m L(f(x^{(i)}), y^{(i)})
  $$
- Com essa escolha de função de perda, a função de custo é convexa, garantindo que o gradiente descendente convergirá para o mínimo global.

### Visualização da Função de Custo
- A função de custo de erro quadrático para classificação resulta em uma superfície irregular, com muitos mínimos locais.
- A nova função de custo para regressão logística produz uma superfície suave e convexa, ideal para otimização.

## Simplificação da Função de Perda e Custo em Regressão Logística

### Revisão da Função de Perda
- A função de perda para regressão logística foi definida como:
  - Se $y = 1$: $$L(f(x), y) = -\log(f(x))$$.
  - Se $y = 0$: $$L(f(x), y) = -\log(1 - f(x))$$.

### Simplificação da Função de Perda
- Como $y$ só pode ser 0 ou 1, a função de perda pode ser escrita de forma compacta:
  $$
  L(f(x), y) = -y \cdot \log(f(x)) - (1 - y) \cdot \log(1 - f(x))
  $$
- **Intuição**:
  - Quando $y = 1$: O segundo termo desaparece, e a função se reduz a $-\log(f(x))$.
  - Quando $y = 0$: O primeiro termo desaparece, e a função se reduz a $-\log(1 - f(x))$.
- Essa fórmula única é equivalente à definição original, mas mais compacta e fácil de implementar.

### Função de Custo Simplificada
- A função de custo $J(w, b)$ é a média das funções de perda sobre todos os exemplos de treinamento:
  $$
  J(w, b) = \frac{1}{m} \sum_{i=1}^m \left[ -y^{(i)} \cdot \log(f(x^{(i)})) - (1 - y^{(i)}) \cdot \log(1 - f(x^{(i)})) \right]
  $$
- Essa é a função de custo comumente usada para treinar modelos de regressão logística.

### Justificativa Estatística
- A função de custo é derivada do princípio de **máxima verossimilhança** (maximum likelihood estimation), um conceito estatístico para encontrar parâmetros de modelos de forma eficiente.
- Propriedade importante: A função de custo é **convexa**, o que garante que o gradiente descendente convergirá para o mínimo global.

# Gradiente Descendente para Regressão Logística

## Implementação do Gradiente Descendente para Regressão Logística

### Objetivo
- Encontrar os parâmetros $w$ e $b$ que minimizam a função de custo $J(w, b)$ usando o **gradiente descendente**.

### Gradiente Descendente para Regressão Logística
- O gradiente descendente atualiza os parâmetros $w$ e $b$ da seguinte forma:
  $$
  w_j := w_j - \alpha \frac{\partial J}{\partial w_j}
  $$
  $$
  b := b - \alpha \frac{\partial J}{\partial b}
  $$
  Onde:
  - $\alpha$: taxa de aprendizado.
  - $\frac{\partial J}{\partial w_j}$: derivada parcial da função de custo em relação a $w_j$.
  - $\frac{\partial J}{\partial b}$: derivada parcial da função de custo em relação a $b$.

### Derivadas Parciais
- Para $w_j$:
  $$
  \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (f(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
  $$
- Para $b$:
  $$
  \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (f(x^{(i)}) - y^{(i)})
  $$
  Onde:
  - $f(x)$: função sigmoid aplicada a $wx + b$.
  - $x_j^{(i)}$: j-ésima característica do i-ésimo exemplo de treinamento.

### Atualização Simultânea
- As atualizações de $w_j$ e $b$ são feitas **simultaneamente**:
  1. Calcula-se o lado direito das equações para todos os parâmetros.
  2. Atualizam-se todos os parâmetros ao mesmo tempo.

### Diferença entre Regressão Linear e Logística
- Embora as equações de atualização sejam semelhantes, a função $f(x)$ é diferente:
  - **Regressão Linear**: $f(x) = wx + b$.
  - **Regressão Logística**: $f(x) = \text{sigmoid}(wx + b)$.
- Portanto, apesar da semelhança nas equações, os algoritmos são fundamentalmente diferentes.

### Aceleração do Gradiente Descendente
- **Escalonamento de características (feature scaling)**:
  - Ajustar as características para que tenham intervalos de valores semelhantes (ex.: entre -1 e 1).
  - Isso ajuda o gradiente descendente a convergir mais rapidamente, tanto para regressão linear quanto para regressão logística.

# O Problema do Overfitting

## **Underfitting** e **Overfitting**

### Introdução
- **Underfitting** e **Overfitting** são problemas comuns em algoritmos de aprendizado de máquina, como regressão linear e regressão logística.
- Esses problemas afetam a capacidade do modelo de generalizar bem para novos dados.
- **Regularização** é uma técnica útil para minimizar o overfitting.

---

### **Underfitting (Alto Viés)**
- Ocorre quando o modelo é muito simples e não consegue capturar padrões nos dados de treinamento.
- Exemplo: Ajustar uma linha reta (regressão linear) a dados que claramente seguem uma tendência não linear.
  - O modelo não se ajusta bem aos dados de treinamento.
  - Termo técnico: **Alto viés (high bias)**.
  - Analogia: O modelo tem uma "preconcepção" forte de que os dados são lineares, ignorando evidências contrárias.

---

### **Overfitting (Alta Variância)**
- Ocorre quando o modelo é muito complexo e se ajusta demais aos dados de treinamento, capturando até o ruído.
- Exemplo: Ajustar um polinômio de alta ordem (ex.: 4ª ordem) a um pequeno conjunto de dados.
  - O modelo passa por todos os pontos de treinamento, mas não generaliza bem para novos dados.
  - Termo técnico: **Alta variância (high variance)**.
  - Analogia: O modelo é muito sensível a pequenas variações nos dados de treinamento, resultando em previsões instáveis.

---

### **Generalização**
- O objetivo do aprendizado de máquina é encontrar um modelo que generalize bem, ou seja, que faça boas previsões em dados nunca vistos antes.
- Um modelo "ideal" não sofre de underfitting nem de overfitting.

---

### Exemplos Práticos

#### **Regressão Linear**
1. **Underfitting**:
   - Modelo linear (reta) em dados que claramente seguem uma tendência não linear.
   - Exemplo: Prever preços de casas com base no tamanho, mas o modelo linear não captura a tendência de saturação dos preços.
2. **Overfitting**:
   - Modelo polinomial de alta ordem (ex.: 4ª ordem) que passa por todos os pontos de treinamento, mas faz previsões irrealistas para novos dados.
   - Exemplo: Uma curva muito "ondulada" que não reflete a tendência real dos preços.

#### **Classificação (Regressão Logística)**
1. **Underfitting**:
   - Modelo linear simples que não consegue separar bem as classes.
   - Exemplo: Uma reta como fronteira de decisão para classificar tumores como malignos ou benignos.
2. **Overfitting**:
   - Modelo complexo com muitos termos polinomiais que cria uma fronteira de decisão muito irregular.
   - Exemplo: Uma fronteira de decisão que se ajusta perfeitamente aos dados de treinamento, mas não generaliza bem.

---

### Analogia: **Cachinhos Dourados e os Três Ursos**
- **Underfitting**: A papa de porridge está muito fria (não serve).
- **Overfitting**: A papa de porridge está muito quente (não serve).
- **Modelo Ideal**: A papa de porridge está na temperatura certa (serve).

---

### Como Evitar Underfitting e Overfitting
1. **Underfitting**:
   - Aumentar a complexidade do modelo (ex.: adicionar mais características ou termos polinomiais).
2. **Overfitting**:
   - Reduzir a complexidade do modelo (ex.: usar menos características ou aplicar regularização).