# Design Note of Parameter System

## Multi-Purpused Parameter System

This Parameter System Design needs to meet the following demands simultaneously.

* Guide the compiling of HSACO kernels from Triton source
* Define `<Interface>Params` structs
* Fill `Functional` related strings in autotune/optune files.

## Roles of Classes

* TemplateParameter: hold metadata of parameter
  - This is abbreviated as "TP" in the document
* TypedChoice: hold choice values along with their metadata
  - This is abbreviated as "TC" in the document
  - All constexpr are stored in a subclass TypedChoice.constexpr, abbreviated as "TCC"
* Bind: describe the association between TemplateParameter and TypedChoice

## Goal

This complicated system is mainly to accomodate two features

1. Tensor with different ranks
2. Conditional value

## Use Case for Tensor

In C++, this feature is similar to:

``` C++
typename<typename T>
void func(Eigen::Tensor<T, 2>& Q, Eigen::Tensor<T, 3>& K);
```

T only defines the elemental type, and the real types of Q and K have to be derived.

In our case, it is

``` python
TYPE_CHOICES = {
    ('Q', 'K') : ['*fp16', '*bf16'],
}
TENSOR_RANKS = {
    'Q' : 2,
    'K' : 3,
}
```

We do not have such situations in current kernels, but this is possible.

## Add ConditionalValue

``` python
TYPE_CHOICES = {
    ('Q', 'K') : ['*fp16', '*bf16'],
    ('V',) : [CDETensor('USE_V', False, 0, 'Q')],
}
TENSOR_RANKS = {
    'Q' : 2,
    'K' : 3,
}
```

The actual type of V is only settled when Q is selected.

## The Process

### Initialization

Let's consider the example above. Note we ignored `USE_V` for simplicity, and
its corresponding TP is denoted as `fp_x` (Feature Parameter X)

```
literal in TYPE_CHOICES
-> (parse_choices)
-> Stored in TemplateParameter.choices.
   Let tp1 = TypeParameter(('Q', 'K'), ['*fp16', '*bf16'])
       tp2 = TypeParameter(('V',), [CDETensor('USE_V', False, 0, 'Q')])
       tp3 = TypeParameter(('HDIM',), [16, 32, 48, 64])
       TC  = typed_choice
       TCC = typed_choice.constexpr
   tp1.choices = [ TC.tensor(elem_ty='fp16', rank=any), TC.tensor(elem_ty='bf16', rank=any) ]
   tp2.choices = [ TC.CDETensor('USE_V', False, 0, 'Q') ] // CDETensor unchanged
   tp3.choices = TypeParameter(('HDIM',), [TCC.int16(16), TCC.int16(32), TCC.int16(48), TCC.int16(64)])
   // Note tp3 (HDIM) is fully resolved (called "settled") now
-> (late_init() -> link_deferral_target())
-> tp2  : [ CDETensor(fp_x, False, 0, tp1) ] // Link CDETensor to tp1 and fp_x (USE_V)
-> (late_init() -> resolve_rank())
## Design I
-> tp1.type_matrix = { // argname -> list of TC
     'Q' : [ TC.tensor(elem_ty='fp16', rank=2), TC.tensor(elem_ty='fp16', rank=2) ]
     'K' : [ TC.tensor(elem_ty='fp16', rank=3), TC.tensor(elem_ty='fp16', rank=3) ] // Note rank=3
   }
   // tp1 is settled now
## Design II
-> tp1.choices = [
        TC.tensor(elem_ty='fp16', rank=any).specialized = {
            'Q' : TC.tensor(elem_ty='fp16', rank=2),
            'K' : TC.tensor(elem_ty='fp16', rank=3),
        },
        TC.tensor(elem_ty='bf16', rank=any).specialized = {
            'Q' : TC.tensor(elem_ty='bf16', rank=2),
            'K' : TC.tensor(elem_ty='bf16', rank=3),
        },
   ]
```

Either with Design I or Design II, the initialization of TPs (TemplateParameter) is complete now.
Some TP (tp2) is not settled but it is okay. ConditionalValue can only be settled within a Functional.

### Settle ConditionalValue

Let's demostrate with a more complicated scenario:

``` python
TYPE_CHOICES = {
    ('Q', 'K') : ['*fp16', '*bf16'],
    ('V',) : [CDETensor('USE_V', False, 0, 'Q')],
    ('Vscale',) : [CC('USE_V', False, 0, 'fp32')],
}
FEAT_CHOICES = {
    ('USE_V',) : [ False, True ]
}
TENSOR_RANKS = {
    'Q' : 2,
    'K' : 3,
}
```

Here we added another parameter "Vscale" which is compiled as `constexpr(0)`
when `USE_V=False`, otherwise it is a `fp32` scalar.

It is not difficult to set the TemplateParameter for `USE_V` will be initialized as
`fp1 = TypeParameter(('USE_V',), [TC.bool_t(True), TC.bool_t(False)])`.

```
-> (itertools.product(*kdesc.list_functional_params()))
   -> (Parameter.__iter__)
      -> (yield Bind(self, CDETensor('USE_V', False, 0, tp1), 0))
-> f1 = Functional(kdesc, .., fbinds, ...)
   Let fbinds = [
                  // Note here the rank is still "any", since Bind only store the
                  // selection of template parameter T. Either tp1.type_matrix
                  // (Design I) or TC.tensor.specialized (Design II) must be
                  // used to settle the type of Q. (This actually is an example
                  // showing Design II is better than I)
                  Bind(tp1, TC.tensor(elem_ty='bf16', rank=any), 1),
                  // Note here nth = 0, referring to CDETensor's index, not tp1's index
                  Bind(tp2, CDETensor(fp1, False, 0, tp1), 0),
                  Bind(tp3, CC(fp1, False, 0, 'fp32'), 0),
                  Bind(fp1, TC.bool_t(True)),
                ]
-> (Functional.__init__ -> __settle_conditional_values)
   -> (Bind.settle_unresolved)
   -> (Replace CDETensor with bind_dict[tp1.repr_name])
      // It seems retriving .repr_name is more verbose than keeping "Q"
      // However tp1 is needed in other scenarios
   -> Bind(tp2, Bind(tp1, TC.tensor(elem_ty='bf16', rank=any), 1), 0),
   -> (Replace Bind(tp1) with Bind(tp1).resolve('Q') )
   -> Bind(tp2, TC.tensor(elem_ty='bf16', rank=2), 0),
```

<del> Note, here a common interface `.resolve(aname, bind_dict)` shared by `Bind` and
`TC` types is needed to settle ConditionalValue in Bind objects. </del>

Bind.resolve is replaced with `.value/.get_typed_value`. The planned `Bind.resolve(aname, bind_dict)`
can be easily replaced with `Bind.value.resolve(aname, bind_dict)`

Hence the end state of `__settle_conditional_values` is 
`Bind(tp2, TC.tensor(elem_ty='bf16', rank=any), 0)` (unranked).
To resolve the ranked TC.tensor object (`TC.tensor(elem_ty='bf16', rank=2)`),
call `Bind(tp2).value.resolve('Q', bind_dict=None)`.

Also, we still keep 'Q' in practice.

## Goal: Guide Compiling of HSACO kernels

The core tasks for this demand is:

1. Determine the triton kernel signature
2. Determine the output file

Task 2 is shared by autotune. This section only consider about the generation of triton kernel signature.

For KernelSignature object, the generation of perf fields and compiler options
are simple, which are simple list with scalar values.

The remaining triton kernel signature can be generated from the linked Functional object:
``` python
for bind in self._binds:
    for aname, atype for bind:
        index = kdesc.ARGUMENTS.index(aname)
        sig_list[index] = atype.triton_compile_signature

Bind.__iter__(self):
    for aname in self._klass.all_names:
        yield aname, self.get_typed_value(aname)
```

To make the code easier to read, Bind is made iterable.
`Bind.get_typed_value` will return the `TC.*` object.

Example `triton_compile_signature` values for `TC.*` objects

|  Typed Choice         |           `triton_compile_signature`            |
|-----------------------|-------------------------------------------------|
|  TCC.int32  2.2       |           (the value)                           |
|  TCC.bool\_t          |           ('True' or 'False')                   |
|  TC.int32             |           'i32'                                 |
|  TC.tensor            |           `'*' + elem_ty`                       |

## Goal: Define `<Interface>Params` structs

This part is a bit more challenging than guiding the compiling process, because
the following objects are unavailable at the moment:

* The real types of arguments. They are unsettled without bindings. This is
  especially True for ConditionalValue.
* The Bind objects. The `<Interface>Params` structs should be uniform for all binds

Therefore, the generation of the struct must only depend on TP objects, and their unsettled TC objects.

The process could be

* Get lists of (itype (Interface Type), aname) from TP objects
    - Use dataclass instead of tuple
    - Maybe call it `cfield`?
* Sort by order in ARGUMENTS
* Print itype aname

Basically a dedicated interface `.get_itype(aname)` is needed for all TC objects.
Specifically, for unsettled TC:

* ConditionalValue should forward the non-optimized version's `.itype`
* TC.tensor should return `.TensorView<{rank}>` regardless of its `elem_ty`
* `aname` is required for `TC.tensor` to resolve the rank

### CDETensor in `<Interface>Params`

In "Settle ConditionalValue" Section, we only discussed the resolution of
CDETensor with Functional context. However, code generation for
`<Interface>Params` does not have such context.

In order to processed, we decided to re-use the TC.tensor object in the
deferred TP to track the default type for code generation

More concretely, it is

```
TYPE_CHOICES = {
    ('Q', 'K') : ['*fp16', '*bf16'],                // tp1
    ('V',) : [CDETensor('USE_V', False, 0, 'Q')],   // tp2
}
FEAT_CHOICES = {
    ('USE_V',) : [ False, True ]                    // fp1
}
TENSOR_RANKS = {
    'Q' : 2,
    'K' : 3,
    'V' : 4,
}

(initialization, but without CDETensor handling)
-> tp1.choices = [
        TC.tensor(elem_ty='fp16', rank=any).specialized = {
            'Q' : TC.tensor(elem_ty='fp16', rank=2),
            'K' : TC.tensor(elem_ty='fp16', rank=3),
        },
        TC.tensor(elem_ty='bf16', rank=any).specialized = {
            'Q' : TC.tensor(elem_ty='bf16', rank=2),
            'K' : TC.tensor(elem_ty='bf16', rank=3),
        },
   ]
   tp2.choices = [
        CDETensor(fp1, False, 0, tp1),
   ]
(CDETensor handling, implemented in initialization, this is to show the delta change)
-> tp1.choices = [
        TC.tensor(elem_ty='fp16', rank=any).specialized = {
            'Q' : TC.tensor(elem_ty='fp16', rank=2),
            'K' : TC.tensor(elem_ty='fp16', rank=3),
            'V' : TC.tensor(elem_ty='fp16', rank=3),
        },
        TC.tensor(elem_ty='bf16', rank=any).specialized = {
            'Q' : TC.tensor(elem_ty='bf16', rank=2),
            'K' : TC.tensor(elem_ty='bf16', rank=3),
            'V' : TC.tensor(elem_ty='bf16', rank=4),
        },
   ]
   tp2.choices = [
        CDETensor(fp1, False, 0, tp1),
   ]
```

To achieve this, `resolve_rank` is added to the base class TypedChoice and defaults to no-op,
and also added to ConditionalDeferredElseTensor (CDETensor) as well.

### Tensor Strides

Tensor Strides should be added to `kdesc._func_params` as TPs.
We may use two TPs to simplify this process ("u64:8" and constexpr(1)).

However, strides are little different from other paramters, they are used to
generate the compiling sigature, but should not present in `<Interface>Params`
structs since they will be supplied along with TensorView objects.

In V2 design, ignoring the fields is hardcoded in `ArgumentMetadata` class
(Similar to TP class in v3), which will return empty if name matches `stride_dtype*`.

In V3, as a more general approach, when inserting TPs to `_func_params` list, TP for strides should be marked as hidden from `.get_itype` (or subclass TP and redefine `get_itype` to empty)

