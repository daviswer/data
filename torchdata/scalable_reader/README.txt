┌─────────────────────────────────────────────────────────────────────────────────┐
│                          RESCALABLE DATALOADER ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────┐
                              │   StatefulDataLoader │
                              │   (PyTorch wrapper)  │
                              └──────────┬──────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   WRAPPERS                                       │
│                           (Composable Pipeline Stages)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  PreprocessDataset → ShuffleDataset → DocPackingDataset → SamplingDataset       │
│                                                                                  │
│  Each wrapper:                                                                   │
│    • Extends _StatefulDataset                                                   │
│    • Tags its state: state_vars | broadcast_vars | reshard_vars | custom_vars   │
│    • Recursively aggregates state_dict() from nested layers                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SCALABLE READER                                     │
│                         (Core Data Orchestrator)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  • Large logical shards (fixed at the start) allocated round-robin to workers   │
│  • ShardStateManager tracks [shard_id, file_pos, doc_pos, chunk_pos, epoch]     │
│  • Fractional file assignment via _get_shard_breakdown()                        │
│  • Shuffle layer decorrelates rank → file mapping                               │
│  • Tags: broadcast_vars=["filesizes"], custom_vars=["shard_states"]             │
└────────────────┬───────────────────────────────────┬────────────────────────────┘
                 │                                   │
                 ▼                                   ▼
┌────────────────────────────────┐    ┌────────────────────────────────────────────┐
│        FILE HANDLERS           │    │                    DCP                      │
│    (Storage Abstraction)       │    │       (Distributed Checkpointing)           │
├────────────────────────────────┤    ├────────────────────────────────────────────┤
│  ShardFileHandler (abstract)   │    │  SAVE:                                      │
│    • is_legal(path)            │    │    • Collect state from all workers         │
│    • open(path)                │    │    • Categorize: state|broadcast|reshard    │
│    • length(path)              │    │    • Wrap reshard_vars in DTensor           │
│    • get(reader, idx)          │    │    • Write to shared storage                │
│    • slice(doc, idx, n)        │    │                                             │
│                                │    │  LOAD (same worldsize):                     │
│  Implementations:              │    │    • Direct load, split to workers          │
│    • ArrowHandler (O(1) access)│    │                                             │
│    • ParquetHandler (+tokenize)│    │  LOAD (rescale):                            │
│                                │    │    • LocalShardsWrapper specifies slice     │
│  Enables:                      │    │    • DCP reads from old checkpoint files    │
│    • Random doc access         │    │    • shard_rescale() for custom resharding  │
│    • Fractional file reads     │    │    • Round-robin reallocation               │
└────────────────────────────────┘    └────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW SUMMARY                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  FORWARD (iteration):                                                            │
│    Files → FileHandler.get() → ScalableReader → Wrappers → DataLoader → Model   │
│                                                                                  │
│  SAVE:                                                                           │
│    All layers' state_dict() → DCP.save() → Shared Storage (per-rank files)      │
│                                                                                  │
│  LOAD (same world):                                                              │
│    Shared Storage → DCP.easy_load() → load_state_dict() → Resume                │
│                                                                                  │
│  LOAD (rescale):                                                                 │
│    Shared Storage → DCP.rescale_load() → DTensor resharding → Resume            │
│                      └→ shard_rescale() for custom vars (order preservation)    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           KEY DESIGN PRINCIPLES                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1. Logical shards >> workers → negligible imbalance, easy rescaling            │
│  2. Round-robin allocation → global order preserved across any world size       │
│  3. State categorization → each var type gets correct rescaling behavior        │
│  4. DTensor abstraction → DCP handles N→M file mapping automatically            │
│  5. Composable wrappers → any pipeline combination is rescalable                │
└─────────────────────────────────────────────────────────────────────────────────┘

