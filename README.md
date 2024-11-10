# DyST-CON: An Efficient Solution for GPUs to the ST-connectivity Problem on Dynamic Graphs

This repository contains the implementation for DyST-CON.

ST-connectivity poses a decision problem, determining whether, for vertices *s* and *t* within a graph, *t* is reachable from *s*. The challenge arises in the context of dynamic real-world graphs that undergo rapid evolution over time. In these scenarios, repeatedly solving the s-t connectivity problem from the beginning after each graph modification becomes impractical. Although parallel solutions, especially designed for GPUs, have been introduced to tackle the size complexity of static graphs, none
have specifically addressed the concern of work efficiency in dynamic graphs. We introduce an efficient solution for GPUs to the st-connectivity problem that can handle concurrent processing of batches of graph updates. Our solution utilizes batch information strategically to reduce the overall workload needed for updating the connectivity result.

## How to run

The code is based on [Hornet](https://github.com/hornet-gt/hornet) (read the [Hornet README.md](https://github.com/PARCO-LAB/DyST-CON/blob/main/cuhornet/README.md) file to understand how to compile it).

The DyST-CON code is implemented by the [STCON.cuh](https://github.com/PARCO-LAB/DyST-CON/blob/main/cuhornet/hornetsnest/include/Dynamic/ST-CON/STCON.cuh) file and can be tested by compiling and running [STCONTest.cu](https://github.com/PARCO-LAB/DyST-CON/blob/main/cuhornet/hornetsnest/test/STCONTest.cu). The test executes both the naive implementation and the proposed solution and compare them.

```
./<executable> <graph_file> batch_size
```

## Publication

The link to the paper will be inserted when available.
