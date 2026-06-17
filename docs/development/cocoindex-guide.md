# CocoIndex Integration & Best Practices

This document outlines the best practices and guidelines for integrating **CocoIndex v1** into the Pocket codebase.

---

## Core Rules for CocoIndex v1

### 1. Never Use v0 APIs
CocoIndex v1 has a completely redesigned API. Do not use any of the following deprecated v0 symbols:
- ❌ `@cocoindex.flow_def`, `FlowBuilder`, `Flow`, `open_flow`
- ❌ `DataScope`, `DataSlice`, `add_collector()`, `collect()`, `export()`
- ❌ `cocoindex.sources.LocalFile`, `cocoindex.sources.*`
- ❌ `cocoindex.functions.SplitRecursively`, `cocoindex.functions.*`
- ❌ `cocoindex.targets.Postgres`, `cocoindex.targets.*`
- ❌ `transform_flow`, `cocoindex.op.function()`

Instead, use the v1 equivalents:
- `coco.App` + a `@coco.fn` main function
- Declare target states via Target APIs (`declare_row`, `declare_file`) inside mounted components
- Connector APIs like `localfs.walk_dir(...)`
- `cocoindex.ops.*` like `RecursiveSplitter`
- Connector targets like `postgres.mount_table_target(...)` or `sqlite.mount_table_target(...)`

### 2. Always Decorate Processing Functions with `@coco.fn`
Every function that participates in the pipeline (declares target states, calls mount APIs, etc.) must be decorated with `@coco.fn`:
```python
@coco.fn
async def my_processor(arg1, target):
    target.declare_row(...)
```

### 3. Use `memo=True` for Expensive Operations
To prevent reprocessing unchanged files and wasting compute/API costs, add `memo=True` to file-level processing functions:
```python
@coco.fn(memo=True)
async def process_file(file, table):
    # This will only run if the file content or the code changes
    ...
```

### 4. Ensure Stable Component Paths
When mounting components, ensure the paths are stable across runs. Do not use object references or list indices as subpaths. Use stable keys like file paths or record IDs:
```python
# Good: Stable identifiers
await coco.mount_each(process_file, files.items(), table)

# Bad: Unstable identifiers
for idx, item in enumerate(items):
    await coco.mount(coco.component_subpath(idx), process_item, item)  # Index changes if items are reordered
```

### 5. Use Context for Shared Resources
Load expensive resources (like database connection pools or embedding models) once in the environment lifespan and retrieve them using `coco.use_context()`:
```python
EMBEDDER = coco.ContextKey[SentenceTransformerEmbedder]("embedder")

@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    builder.provide(EMBEDDER, SentenceTransformerEmbedder("all-MiniLM-L6-v2"))
    yield

@coco.fn
async def process_chunk(chunk, table):
    embedder = coco.use_context(EMBEDDER)
    embedding = await embedder.embed(chunk.text)
    ...
```
