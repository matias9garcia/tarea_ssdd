# Load train_input y train_label
# Load param_config
# # Data entrada : Xe(d, N), d: entrada, N: muestras
# # Data salida : Ye(nC, N), nC: número de clases
# # Nodos Ocultos: L, Penalidad Pinv: C
# [w1 bias w2] = upd_pesos(Xe, Ye, L, C)
# #costo=calc_costo(Xe,Ye,w1,bias,w2)
# #Grabar pesos del IDS
# save(‘pesos’, w1, bias, w2)
# #Grabar vector de Costo
# #savetxt( ‘costo.csv’, costo)


# Function upd_date(xe, ye, L, C)
# [Dim N] = size(xe)
# w1 = rand_W(L, Dim)  # Weigth hidden
# bias = rand_W(L, 1)  # Bias hidden
# biasMatrix = repmat(bias, 1, N)  # bias Matrix
# z = w1*xe + biasMatrix
# H = Activation(z)
# #-Calculate output weights
# yh = ye*H'
# hh = (H*H‘+ eye(L)/C)
# inv = pinv(hh)
# w2 = yh*inv
# return(w1, bias, w2)
# Endfunction


# Function[zv] = forward(xv, w1, bias, w2)
# [D N] = size(xv)
# biasMatrix = repmat(bias, 1, N)
# z = w1*xv + biasMatrix
# H = Activation(z)
# z = w2*H
# return(z)
# EndFunction
