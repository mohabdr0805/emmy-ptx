# emmy-ptx

IR to PTX kernel compiler for Emmy hackathon, made in Python.

    python compile.py path/to/kernel.ir.json > out.ptx

Only `emmy-ml==0.3.11` is needed to read the IR with `Graph.from_dict`. All the PTX is written with `compile.py`, without any other translator (Emmy or nvcc).


## Results

- All 15 open problems work, cat 1 to 5.
- From the 51 variants from `tools/generate.py` (other sizes, other ops, f16, broadcasting, softmax, layernorm, matmul) : 50 work. The one fail needs `Select`.
- Duration is about 1.08 times Emmy's time, on the geometric mean of 24 problems, of which 20 on par.

## How it is built

The compiler goes through the IR tree with a method for each node type. An unknown node raises an error. 

What it does is mainly translation, Emmy already chooses tiling, unrolling and access width, and ptxas does the register allocations; so almost no optimizations.
Registers are virtual and new, except for a name that the IR reassigns; it keeps its register, otherwise the value would be lost after a loop or an if condition.

Few choices not in the IR :
- the guard `gid < N` is only emitted when the launch has more threads than units, because a thread that exits early doesn't go through `bar.sync` anymore
- arithmetic around index is unsigned, so a division by a power of 2 becomes a shift
- index and addresses already computed are reused, and its range stop on loops and if conditions
- an inner loop marked unroll is unrolled at compile time, and a long inner loop without barriers is unrolled 8 by 8 to keep several loads in flight
- f16 is stored as f16 and computed as f32, so only rounded once on store
- beyond 48 KB of shared memory, arrays go to a dynamic pool (`smem_bytes` in `meta.json`)
- without `meta.json`, parameter order and shapes are read in the IR and the guard is always emitted

## Limits

- `Select` is not managed, but TASK.md says it appears in no open or hidden problem
- `erf` doesn't have a PTX instruction, so the Abramowitz and Stegun approximation was used (error < 4e-7 measured)
- For the attention, Emmy loads `q` only once before the keys loop, and we read again for every key (a02 : 56 µs against 29). 
  Hoisting this loop invariant means moving loads, so I left it out before the deadline because it needed more testing.
  