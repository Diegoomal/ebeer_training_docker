# exemplo implementação docker

Primeiros passos com **docker** e **python**

## Execução

### Compilação do projeto

 ```
docker image build -t img_app .
```

### Execução do projeto

 ```
docker run --gpus all img_app
```