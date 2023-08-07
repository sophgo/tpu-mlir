# Rewriter design

If the shape is empty, then this type is unranked.

- operation.getOperand(0) -> Value
- operation.getResult(idx) -> Value
- operation.getResults(idx) -> Value
- operation.getElementType(v: Value) -> DType ?
- operation.erase()
- operation.setAttr()/.getAttr()
- operation.moveAfter(otherOperation)
- operation.setLoc(Location())
- newType = Type(shape, dtype)
- operation.setOperand(idx, v: Value)
- operation.setOperands([v1, v2])


- rewriter.setInsertionPointAfter(operation)
- rewriter.replaceAllUsesWith(value, value)
- rewriter.replaceAllUsesExcept(value, value, operation)
- rewriter.create(operation) // loc, result_type, input_value, attrs

```python
@OpRewritePattern("top.MatMulOp")
class ConvertGLMTilePermute:
  @staticmethod
  def matchAndRewrite(op: Operation, rewriter: PatternRewriter) -> LogicalResult:
    ...


```

```py
attrs = Attributes()
attrs['kernel_shape'] = [1,1]
...
```


# Shape Infer Design

```python
# common/{op}.py

class AddPasses(Passes):

  def init(...):
    pass

  def inference(p: Params):
    pass

  def shape_infer(op: Operation):
    ...


```


# Lowering


