# Write-Ahead Logging

Write ahead log persistence. A write ahead log appends every change before it
reaches the table store, so the engine can replay the log and rebuild committed
state after a restart. The write ahead log is how the storage layer stays
consistent.
