# Notes — mini (machine-id d81ba182)

## Tiptoe-go (Go reference) intentionally omitted

The post-2026-05-11 mini suite does **not** include `tiptoe-go` runs
on this machine. `analysis/tiptoe_diff.py` consequently cannot run
on mini; the validation gate is satisfied on machines that do have a
paired run (sacs006 is the canonical pair source).

### Why

The pinned `henrycg/simplepir` revision
(`026ee7bd6783f9d3b8f6fa33abdafe2b30a21e87`) defines methods on CGO
type aliases (`type Elem32 = C.Elem32; func (Elem32) Bitlen()…`).
Go 1.21+ tightened the rule against defining methods on non-local
types; mini's toolchain (`go version go1.26.2 darwin/arm64`) rejects
the build with:

```
./matrix/matrix.go:43:7: cannot define new methods on non-local type Elem32
./matrix/matrix.go:47:7: cannot define new methods on non-local type Elem64
```

Existing tiptoe-go runs were built with an older Go (likely 1.20 in
the original deploy/Dockerfile chain). Patching the vendored
`tools/tiptoe-go.patch` to convert the aliases to type definitions
would change the comparability anchor — `tiptoe-go` is the
*reference* against which the Rust port is validated; the vendored
patch should track the upstream Go ref byte-for-byte except for
strictly additive `paired-runner` glue.

### Validation by transitivity

The Plan 21 B.1 bit-equality unit test
(`scorer-tiptoe::pir::simplepir::tests::encrypt_lhe_matches_materialised_a`,
landed at commit `1cdecc5`) asserts byte-for-byte ciphertext equality
between the new streaming `encrypt_lhe` and the pre-fix
`expand_a` + `lwe::encrypt_vec` path. The pre-fix path had passed
`tiptoe_diff.py` on sacs006 prior to the fix. By transitivity (new
== old, old == Go-ref on sacs006), the new path matches the Go-ref
at the ciphertext level, which propagates through the unchanged
`server_answer` → `client_recover` pipeline to identical top-k IDs.

### Future sacs006 work — caveat

When sacs006 next runs the eval-suite, **the post-fix `tiptoe` Rust
run and the paired `tiptoe-go` run must land in the same suite
invocation**. Don't pair a post-fix Rust run against the existing
pre-fix sacs006 `tiptoe-go` rows under `4d874634/9d238261…/`,
`4d874634/adf5e7b8…/`, `4d874634/939e6379…/` — those were built
against the pre-streaming `encrypt_lhe` and would surface as
spurious differences. Fresh paired runs only.

`analysis/report.py` for mini will skip the tiptoe-go column /
row in figures that compare Rust ↔ Go side-by-side; report
generation continues with the other scorers unchanged.
