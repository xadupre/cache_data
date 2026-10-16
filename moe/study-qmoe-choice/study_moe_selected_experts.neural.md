# QMoE routing results

## Configuration

- Optimization: `neural`
- Device: `GPU (NVIDIA H200)`
- Maximum epochs: `200`
- Learning rate: `0.0005`
- Hidden size: `256`
- Retention lambda: `100.0`
- Early-stopping relative tolerance: `0.0001`
- Early-stopping patience: `20`
- Network per QMoE: `Linear(768, 256, bias=False) + Sigmoid + Linear(256, 256, bias=False) + Sigmoid`

```text
H[t,m,:] = sigmoid(W1[m] @ concat(P[t-1,m,:], S[t-1,m,:], S[t,m-1,:]))
P[t,m,:] = sigmoid(W2[m] @ H[t,m,:])

loss = mean(t,m) sum(i) (P[t,m,i] - S[t,m,i])**2
     + mean(t,m) sum(i,k=1..96) 100.0/(k+100.0) * S[t-k,m,i] * (P[t,m,i] - S[t-k,m,i])**2
```

