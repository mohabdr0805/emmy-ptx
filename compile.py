# Emmy Kernel IR -> PTX.
#   python compile.py path/to/kernel.ir.json > out.ptx

import json, math, struct, sys
from pathlib import Path

from emmy.compiler.graph import Graph

# One PTX instruction per elementwise op. Op names are numpy's.
F32_OPS = {
    "add": "add.f32", "subtract": "sub.f32", "multiply": "mul.f32",
    "divide": "div.rn.f32", "maximum": "max.f32", "minimum": "min.f32",
    "negative": "neg.f32", "abs": "abs.f32", "fabs": "abs.f32",
    "sqrt": "sqrt.rn.f32", "rsqrt": "rsqrt.approx.f32",
    "reciprocal": "rcp.rn.f32", "tanh": "tanh.approx.f32", "copy": "mov.f32",
}
INT_OPS = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32", "/": "div.s32", "%": "rem.s32"}
CMP_OPS = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
NBYTES = {"f32": 4, "float": 4}
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


# Exact bit pattern, no decimal rounding.
def f32_imm(x):
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


class Regs:
    KINDS = {"p": ".pred", "r": ".b32", "rd": ".b64", "f": ".f32"}

    def __init__(self):
        self.count = {k: 0 for k in self.KINDS}

    def new(self, kind):
        self.count[kind] += 1
        return f"%{kind}{self.count[kind]}"

    def decls(self):
        return [f"        .reg {t:<6} %{k}<{self.count[k] + 1}>;"
                for k, t in self.KINDS.items() if self.count[k]]


class Compiler:
    def __init__(self, kernel, launch):
        self.kernel = kernel
        self.launch = launch                # meta.json entry: arg_order, grid, block, args
        self.regs = Regs()
        self.lines = []
        self.smem = []
        self.n_labels = 0
        self.env = {}                       # IR name -> register, or int if known
        self.base = {}                      # buffer -> base address register
        self.space = {}                     # buffer -> "global" or "shared"
        self.shape = {a["name"]: a["shape"] for a in launch["args"]}

    def op(self, opcode, *args):
        self.lines.append(f"        {opcode:<20}{', '.join(map(str, args))};")

    def bra(self, label, pred=""):
        self.lines.append(f"        {pred + ' ' if pred else ''}bra {label};")

    def place(self, label):
        self.lines.append(label + ":")

    def label(self, stem):
        self.n_labels += 1
        return f"{stem}_{self.n_labels}"

    # ---- integers: indices, bounds, predicates

    def int_bin(self, o, a, b):
        if isinstance(a, int) and isinstance(b, int):
            return {"+": a + b, "-": a - b, "*": a * b, "/": a // b, "%": a % b}[o]

        # x + 0 and x * 1 come from the unrolled bodies.
        if (o in "+-" and b == 0) or (o == "*" and b == 1):
            return a

        # PTX takes the immediate second.
        if o in "+*" and isinstance(a, int):
            a, b = b, a
        if isinstance(a, int):
            a = self.mov_int(a)

        r = self.regs.new("r")
        self.op(INT_OPS[o], r, a, b)
        return r

    def mov_int(self, v):
        r = self.regs.new("r")
        self.op("mov.s32", r, v)
        return r

    def int_expr(self, e):
        if isinstance(e, int):
            return e
        t = type(e).__name__
        if t == "Literal":
            return int(e.value)
        if t == "Dim":
            return self.int_expr(e.expr)
        if t == "Var":
            return self.env[e.name]
        if t == "BinaryExpr" and e.op in INT_OPS:
            return self.int_bin(e.op, self.int_expr(e.left), self.int_expr(e.right))
        raise NotImplementedError(f"int expression {e}")

    def pred_expr(self, e):
        if type(e).__name__ != "BinaryExpr" or e.op not in CMP_OPS:
            raise NotImplementedError(f"predicate {e}")

        a, b = self.int_expr(e.left), self.int_expr(e.right)
        if isinstance(a, int):
            a = self.mov_int(a)

        p = self.regs.new("p")
        self.op(f"setp.{CMP_OPS[e.op]}.s32", p, a, b)
        return p

    # ---- memory

    # Row-major against the buffer shape. Empty index = scalar = 0.
    def flat_index(self, buf, index):
        if len(index) == 1:
            return self.int_expr(index[0])

        flat, stride = 0, 1
        for i, n in reversed(list(zip(index, self.shape[buf]))):
            flat = self.int_bin("+", flat, self.int_bin("*", self.int_expr(i), stride))
            stride *= n
        return flat

    # Returns (register, byte offset). Shared addresses are 32-bit.
    def address(self, buf, flat):
        if isinstance(flat, int):
            return self.base[buf], 4 * flat

        if self.space[buf] == "shared":
            a = self.regs.new("r")
            self.op("mad.lo.s32", a, flat, 4, self.base[buf])
            return a, 0

        off, a = self.regs.new("rd"), self.regs.new("rd")
        self.op("mul.wide.s32", off, flat, 4)
        self.op("add.s64", a, self.base[buf], off)
        return a, 0

    def ld(self, dst, buf, addr):
        reg, off = addr
        self.op(f"ld.{self.space[buf]}.f32", dst, f"[{reg}+{off}]")

    def st(self, buf, addr, v):
        reg, off = addr
        self.op(f"st.{self.space[buf]}.f32", f"[{reg}+{off}]", v)

    # A name assigned again (accumulator, shuffle partner) keeps its register,
    # otherwise the new value is lost at the end of an if or a loop body.
    def reg_of(self, name):
        if name not in self.env:
            self.env[name] = self.regs.new("f")
        return self.env[name]

    def elementwise(self, op, d, args):
        if op in F32_OPS:
            self.op(F32_OPS[op], d, *args)
        elif op == "relu":
            self.op("max.f32", d, args[0], f32_imm(0.0))
        elif op in ("exp", "exp_fast"):
            # e^x = 2^(x * log2(e))
            self.op("mul.f32", d, args[0], f32_imm(LOG2E))
            self.op("ex2.approx.f32", d, d)
        elif op == "log":
            self.op("lg2.approx.f32", d, args[0])
            self.op("mul.f32", d, d, f32_imm(LN2))
        else:
            raise NotImplementedError(f"elementwise op {op}")

    # ---- statements

    def stmt(self, s):
        getattr(self, "stmt_" + type(s).__name__)(s)

    def body(self, stmts):
        for s in stmts:
            self.stmt(s)

    def stmt_Tile(self, s):
        tid, ctaid, ntid, gid = (self.regs.new("r") for _ in range(4))
        self.op("mov.u32", tid, "%tid.x")
        self.op("mov.u32", ctaid, "%ctaid.x")
        self.op("mov.u32", ntid, "%ntid.x")
        self.op("mad.lo.s32", gid, ctaid, ntid, tid)

        # lane and warp are used by the IR but never declared.
        lane, warp = self.regs.new("r"), self.regs.new("r")
        self.op("and.b32", lane, tid, 31)
        self.op("shr.u32", warp, tid, 5)
        self.env["lane"], self.env["warp"] = lane, warp

        # Guard only if there are more threads than cells: a thread that exits
        # early never reaches bar.sync and the rest of the block waits forever.
        extents = [self.int_expr(a.extent) for a in s.axes]
        cells = math.prod(extents)
        threads = math.prod(self.launch["grid"]) * math.prod(self.launch["block"])
        if threads > cells:
            p = self.regs.new("p")
            self.op("setp.ge.s32", p, gid, cells)
            self.bra("DONE", f"@{p}")

        # Last axis fastest: (a0, a1_co) = (gid / 128, gid % 128).
        rest = gid
        for axis, n in reversed(list(zip(s.axes[1:], extents[1:]))):
            self.env[axis.name] = self.int_bin("%", rest, n)
            rest = self.int_bin("/", rest, n)
        self.env[s.axes[0].name] = rest

        self.body(s.body)

    def stmt_Load(self, s):
        assert s.dtype.name == "f32", s.dtype.name
        dst = [self.reg_of(n) for n in s.names]
        reg, off = self.address(s.input, self.flat_index(s.input, s.index))

        # Several names = consecutive values = one vector load.
        if len(dst) == 1:
            self.ld(dst[0], s.input, (reg, off))
        else:
            self.op(f"ld.{self.space[s.input]}.v{len(dst)}.f32", "{" + ", ".join(dst) + "}", f"[{reg}+{off}]")

    def stmt_Assign(self, s):
        assert s.dtype is None or s.dtype.name == "f32", s.dtype
        args = [self.env[a] for a in s.args]
        self.elementwise(s.op.name, self.reg_of(s.name), args)

    def stmt_Write(self, s):
        assert s.value_dtype.name == "f32", s.value_dtype.name
        vals = [self.env[v] for v in s.values]
        reg, off = self.address(s.output, self.flat_index(s.output, s.index))

        if s.atomic:
            for k, v in enumerate(vals):
                self.op("red.global.add.f32", f"[{reg}+{off + 4 * k}]", v)
        elif len(vals) == 1:
            self.st(s.output, (reg, off), vals[0])
        else:
            # buf[i] = (v0, v1, v2, v3) = one vector store.
            self.op(f"st.{self.space[s.output]}.v{len(vals)}.f32", f"[{reg}+{off}]", "{" + ", ".join(vals) + "}")

    def stmt_StridedLoop(self, s):
        # Accumulators start at the op's identity, before the loop.
        for acc in accums_of(s.body):
            init = self.env[acc.base] if acc.base else f32_imm(acc.op.identity)
            self.op("mov.f32", self.reg_of(acc.name), init)

        end = self.int_expr(s.end if s.end is not None else s.axis.extent)
        a = self.regs.new("r")
        self.op("mov.s32", a, self.int_expr(s.start))
        self.env[s.axis.name] = a

        top, done = self.label("LOOP"), self.label("LOOP_END")
        p = self.regs.new("p")
        self.place(top)
        self.op("setp.ge.s32", p, a, end)
        self.bra(done, f"@{p}")

        self.body(s.body)

        self.op("add.s32", a, a, self.int_expr(s.step))
        self.bra(top)
        self.place(done)

    def stmt_Accum(self, s):
        acc = self.env[s.name]
        self.elementwise(s.op.name, acc, [acc, self.env[s.value]])

    def stmt_WarpShuffle(self, s):
        # XOR butterfly: every lane ends with the result, not only lane 0.
        # The combine is the IR's own body, so max works like add.
        step = s.length // 2
        while step:
            for st, sb in zip(s.state, s.state_b):
                self.op("shfl.sync.bfly.b32", self.reg_of(sb), self.env[st], step, "0x1f", "0xffffffff")
            self.body(s.combine_states)
            step //= 2

    def stmt_Smem(self, s):
        extents = [self.int_expr(e) for e in s.extents]
        nbytes = NBYTES[s.dtype] * math.prod(extents)
        self.smem.append(f".shared .align {max(s.align, 16)} .b8 {s.name}[{nbytes}];")

        self.shape[s.name] = extents
        self.space[s.name] = "shared"
        self.base[s.name] = self.regs.new("r")
        self.op("mov.u32", self.base[s.name], s.name)

    def stmt_Cond(self, s):
        p = self.pred_expr(s.cond)
        skip, end = self.label("ELSE"), self.label("END_IF")
        self.bra(skip, f"@!{p}")

        self.body(s.body)

        if s.else_body:
            self.bra(end)
            self.place(skip)
            self.body(s.else_body)
            self.place(end)
        else:
            self.place(skip)

    def stmt_Sync(self, s):
        if s.warp:
            self.op("bar.warp.sync", "0xffffffff")
        elif s.count:
            self.op("bar.sync", s.barrier_id, s.count)
        else:
            self.op("bar.sync", s.barrier_id)

    def stmt_TreeHalve(self, s):
        if s.inner is not None:
            raise NotImplementedError("TreeHalve with inner")

        tid = self.env[s.tid_var]
        bar = (s.barrier_id, s.barrier_count) if s.barrier_count else (s.barrier_id,)

        # Unrolled, the length is static. Lower half folds in the upper half.
        half = s.length // 2
        while half:
            p = self.regs.new("p")
            skip = self.label("TREE")
            self.op("setp.lt.s32", p, tid, half)
            self.bra(skip, f"@!{p}")

            for buf, st, sb in zip(s.bufs, s.state, s.state_b):
                self.ld(self.reg_of(st), buf, self.address(buf, tid))
                self.ld(self.reg_of(sb), buf, self.address(buf, self.int_bin("+", tid, half)))
            self.body(s.combine_states)
            for buf, st in zip(s.bufs, s.state):
                self.st(buf, self.address(buf, tid), self.env[st])

            self.place(skip)
            self.op("bar.sync", *bar)
            half //= 2

        # Every thread reads the result.
        for buf, st in zip(s.bufs, s.state):
            self.ld(self.env[st], buf, self.address(buf, 0))

    def compile(self):
        # Params are generic pointers, converted once.
        for arg in self.launch["arg_order"]:
            p, g = self.regs.new("rd"), self.regs.new("rd")
            self.op("ld.param.u64", p, f"[{arg}]")
            self.op("cvta.to.global.u64", g, p)
            self.base[arg], self.space[arg] = g, "global"

        self.body(self.kernel.body)
        self.place("DONE")
        self.lines.append("        ret;")

        params = ",\n".join(f"        .param .u64 {a}" for a in self.launch["arg_order"])
        return "\n".join([
            ".version 8.0", ".target sm_89", ".address_size 64", "",
            *self.smem, "",
            f".visible .entry {self.kernel.name}(", params, ")", "{",
            *self.regs.decls(), "",
            *self.lines, "}", "",
        ])


# Accumulators of a loop body. A nested loop seeds its own.
def accums_of(stmts):
    for s in stmts:
        t = type(s).__name__
        if t == "Accum":
            yield s
        elif t == "Cond":
            yield from accums_of(s.body)
            yield from accums_of(s.else_body)


def load(path):
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    (kernel,) = [n.op for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]

    # Param order comes from meta.json, not from the IR node.
    meta = json.loads((path.parent / "meta.json").read_text())
    (launch,) = [k for k in meta["kernels"] if k["entry"] == kernel.name]
    return kernel, launch


if __name__ == "__main__":
    kernel, launch = load(sys.argv[1])
    sys.stdout.write(Compiler(kernel, launch).compile())
