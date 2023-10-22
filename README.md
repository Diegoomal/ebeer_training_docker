# exemplo implementação docker

Primeiros passos com **docker**, **python**, **tensorflow**, **GPU(NVIDIA)**, 

## Passos para compilação e execução

### Compilação do projeto

 ```
docker image build -t img_app_ebeer .
```

### Execução do projeto

 ```
docker run --name container_app_ebeer --gpus all img_app_ebeer
```