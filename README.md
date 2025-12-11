# Classificação de Similaridade Textual com BERT — Dataset ASSIN2

Este projeto utiliza o modelo **BERTimbau Base** para realizar **classificação binária de similaridade textual** no dataset **ASSIN2**, prevendo se dois textos são semanticamente semelhantes (`1`) ou não (`0`).

---

## Dataset — ASSIN2

O dataset **ASSIN2 (Avaliação de Similaridade e Inferência Textual)** contém pares de frases em português com rótulos binários indicando similaridade textual.

No projeto:

- **6.500 amostras de treino**
- Dataset **balanceado**:  
  - `1`: 3.250 exemplos  
  - `0`: 3.250 exemplos  

Carregado diretamente via **HuggingFace Datasets**, garantindo padronização e reprodutibilidade.

---



## Arquitetura do Modelo

O modelo utilizado foi:

neuralmind/bert-base-portuguese-cased


Implementado via `BertForSequenceClassification`, com:

- Embeddings BERT pré-treinados
- Camada de classificação com 2 saídas (inicializada do zero)
- Otimizador **AdamW**

Aviso esperado exibido durante o carregamento:

```
Some weights ... are newly initialized: ['classifier.bias', 'classifier.weight']
```


 Isso é normal — significa que a camada final foi criada do zero e será treinada.

---

## Processo de Treinamento

Foram realizadas **5 execuções**.

Cada run utilizou:

- **3 épocas**
- Treino feito em **CPU**
- Tempo médio: ~19min por época
- Cálculo de:
  - Loss por época
  - Validation Accuracy
  - Test Accuracy
  - F1-score

---

## Perda por época

Todas as execuções seguiram o mesmo padrão de convergência:

| Época | Loss Médio |
|------|------------|
| 1    | ~0.35      |
| 2    | ~0.17      |
| 3    | ~0.10      |

O modelo converge rapidamente e de forma estável.

---

##  Resultados por Execução

### **Run 1**
- `Val Acc`: **0.9560**
- `Test Acc`: **0.8942**

### **Run 2**
- `Val Acc`: **0.9600**
- `Test Acc`: **0.8954**

### **Run 3**
- `Val Acc`: **0.9560**
- `Test Acc`: **0.8913**

### **Run 4**
- `Val Acc`: **0.9460**
- `Test Acc`: **0.8766**

### **Run 5**
- `Val Acc`: **0.9580**
- `Test Acc`: **0.8922**

---

## Médias Finais

| Métrica       | Valor Médio |
|---------------|-------------|
| **Val Acc**   | **0.9552**  |
| **Val F1**    | **0.9552**  |

Excelente consistência entre as runs, mostrando que o pipeline é robusto.

---

## Exportação dos Resultados

Os resultados foram salvos em:

outputs/resultados_assin2.csv


O arquivo contém:

- Acurácia por run
- F1-score por run
- Médias finais  
- Ótimo para documentação e reprodutibilidade

---

## Conclusões

- O **BERTimbau** apresentou desempenho excepcional na tarefa de Similaridade Textual.
- O dataset ASSIN2, mesmo sendo pequeno, produziu métricas acima de **95% de F1-score**.
- O modelo treinou de forma estável em CPU.
- As 5 execuções mostram consistência e baixo desvio entre métricas.

---

