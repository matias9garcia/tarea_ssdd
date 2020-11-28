# ACTIVIDAD 3

------

## DATA 

**Training: KDDTrain.txt** 

- ​	N-filas: número de muestras.

-  D-columnas: 
  - Las primeras 41-columnas denotas los atributos del clasificador.

  -  La columna 42 representa la etiqueta de la clase: 
    - Normal (N) 
    - Ataque (A)

**Testing: KDDTest.txt**

ES LO MISMO SIN LA ETIQUETA DE CLASE ??



------

## PREPROCESO (preproceso.py)



-  Transformar cada atributo (1-41) a formato numérico. Transformar atributo 42 a clase bipolar. Esto es: 
  -  Clase Normal: 1
  -  Clase Ataque: -1

![image-20201120204611194](C:\Users\arval\AppData\Roaming\Typora\typora-user-images\image-20201120204611194.png)




------



## ENTRAMIENTO (train.py)

**Parámetros:** 

-  Nodos Ocultos :

-  Máx. Iteraciones : 

-  Número de Partículas :

-   Penalidad Pseudo-inversa : 

**Archivos de Salida:**

- Pesos ocultos y pesos de salida 
- Error cuadrático medio vs Iteraciones: 
  -  costo.csv



###  PSEUDOCODIGO 



```python
Load train_input y train_label
Load param_config
# Data entrada : Xe(d, N), d: entrada, N: muestras
# Data salida : Ye(nC, N), nC: número de clases
# Nodos Ocultos: L, Penalidad Pinv: C
[w1 bias w2] = upd_pesos(Xe, Ye, L, C)
#costo=calc_costo(Xe,Ye,w1,bias,w2)
#Grabar pesos del IDS
save(‘pesos’, w1,bias,w2)
#Grabar vector de Costo
#savetxt( ‘costo.csv’, costo)

```

```python
Function upd_date(xe,ye, L, C)
[Dim N] = size(xe)
w1 = rand_W(L, Dim) ; # Weigth hidden
bias = rand_W(L,1) # Bias hidden
biasMatrix = repmat(bias,1,N) # bias Matrix
z = w1*xe +biasMatrix
H = Activation (z)
#-Calculate output weights
yh = ye*H';
hh = (H*H‘+eye(L)/C);
inv = pinv(hh);
w2 = yh*inv;
return(w1, bias, w2)
Endfunction
```




------

## rand_w

### PSEUDOCODIGO

 

```python
Function rand_W(next_nodes,current_nodes)
#W-random into [-r,r]
r=sqrt(6/(next_nodes + current_nodes))
w=rand(next_nodes,current_nodes)*2*r-r;
return(w)
Endfunction
```



------

## PRUEBA (test.py)

 Archivos de Salida: 

- ​	metrica.csv: 
  -  Exactitud 
  -  F-score clase normal 
  -  F-score clase ataque



### PSEUDOCODIGO 

```python
Load test_input y test_label # Data de testing
Load pesos # Pesos estrenados
# Calcular data de salida del IDS
z=forward(Xv, w1,bias,w2)
# Calcular métrica de rendimiento
[accuracy, Fscore]=metrica(z, Yv)
#Grabar Exactitud y r F-score de cada Clase
savetxt( ‘fscore.csv’, Accuracy, Fscore)
```

------

## FORWARD

### PSEUDOCODIGO 

```python
Function [zv]=forward(xv, w1, bias, w2)
[D N] = size(xv);
biasMatrix = repmat(bias,1,N)
z = w1*xv +biasMatrix
H = Activation(z);
z=w2*H;
return(z)
EndFunction
```

## FUNCION DE ACTIVACIÓN

![image-20201127233938480](C:\Users\arval\AppData\Roaming\Typora\typora-user-images\image-20201127233938480.png)

z -> norma uclediana de (X y W)

X -> la entrada (matriz) 

W-> los pesos

##  QPSO (pesos ocultos de la red ) 

