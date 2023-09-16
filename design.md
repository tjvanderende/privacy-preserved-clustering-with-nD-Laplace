::: mermaid
graph TD;
    A[Input X] -->|Run| B(Determine implementation);
    B --> C{Check dimensions};
    C -->|2 dimensions| D[2D-Laplace]
    C -->|3 dimensions| E[3D-Laplace]
    C -->|n dimensions| F[nD-Laplace]
    E --> G[kd-rree truncation]
    D --> G[kd-rree truncation]
    F --> G[kd-rree truncation]
    G --> H[Report Z]
:::
  