import test from "node:test";
import assert from "node:assert/strict";
import { planMissingPageIds } from "../knowledge_sync_plan.mjs";

test("selects only IDs absent from inference cache", () => {
  assert.deepEqual(
    planMissingPageIds(["a", "b", "c", "d"], new Set(["a", "c"]), 5),
    ["b", "d"]
  );
});

test("keeps stable order and respects batch size", () => {
  assert.deepEqual(
    planMissingPageIds(["z", "y", "x", "w"], new Set(), 2),
    ["z", "y"]
  );
});

test("deduplicates approved IDs", () => {
  assert.deepEqual(
    planMissingPageIds(["a", "a", "b", "b", "c"], new Set(["a"]), 5),
    ["b", "c"]
  );
});

test("does not schedule any page already present in cache", () => {
  assert.deepEqual(
    planMissingPageIds(["a", "b"], new Set(["a", "b"]), 5),
    []
  );
});
