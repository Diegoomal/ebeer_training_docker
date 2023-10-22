# exemplo implementação docker

Primeiros passos com **docker**, **python**, **tensorflow**, **GPU(NVIDIA)**, 

## Passos para compilação e execução

### Compilação do projeto

 ```
docker image build -t img_app .
```

### Execução do projeto

 ```
docker run --gpus all img_app
```