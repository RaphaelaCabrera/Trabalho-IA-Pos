# Trabalho-IA-Pos
Trabalho prático da disciplina de Inteligência Artificial do curso de pós-graduação em Ciências da Computação.

O código implementado nesse repositório teve como objetivo treinar cinco algoritmos de classificação
para cinco datasets diferentes com o intuito de comparar o desempenho de cada um para cada 
conjunto de dados.

Os algoritmos escolhidos foram:
- Regressão Logística
- KNN
- SVM
- Árvores de decisão
- Random Forest

Foram selecionados os seguintes datasets relacionados à área da saúde:
- [Doenças cardíacas](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?select=heart_disease_uci.csv)
- [Derrame cerebral](https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset)
- [Mortalidade de câncer de pulmão](https://www.kaggle.com/datasets/masterdatasan/lung-cancer-mortality-datasets-v2)
- [Estimativa dos níveis de obesidade com base nos hábitos alimentares e na condição física](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- [Recorrência de câncer de tireoide](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence)

É necessário instalar as bibliotecas utilizadas. Para isso, execute o comando:
```
pip install -r requirements.txt
```

Para executar o código, basta rodar o comando:
```
python run.py dataset_desejado
```

Onde dataset_desejado corresponde ao dataset a ser utilizado. Esse parâmetro pode ter os seguintes valores:
- dataset1 - Doenças cardíacas
- dataset2 - Derrame cerebral
- dataset3 - Mortalidade de câncer de pulmão
- dataset4 - Estimativa dos níveis de obesidade com base nos hábitos alimentares e na condição física
- dataset5 - Recorrência de câncer de tireoide

